"""
Streamed Inference Manager (Chat Version) — AirLLM + TurboQuant pipeline.
Bu versiyon, gerçek bir sohbet (chat) döngüsü ve autoregressive text generation içerir.
"""

from __future__ import annotations
import gc
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import torch
from transformers import AutoTokenizer

from turboquant.llmtuning_bridge import LLMTuningTurboSession, CachePolicy
from turboquant.kv_cache import KVCacheCompressor
from turboquant.gguf_reader import GGUFMap

# ---------------------------------------------------------------------------
# MATH UTILITIES (TORCH ACCELERATED)
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: Root Mean Square Layer Normalization."""
    # x: (batch, seq, dim)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

def precompute_rope_freqs(dim: int, end: int, theta: float = 1000000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE (Rotary Positional Embedding) frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Query or Key tensors."""
    # x: (batch, n_heads, seq, head_dim)
    # cos/sin: (seq, head_dim // 2)
    x_real = x[..., 0::2]
    x_imag = x[..., 1::2]
    
    # cos/sin shape adjustment for broadcasting
    cos = cos[:x.shape[2], :].view(1, 1, x.shape[2], -1)
    sin = sin[:x.shape[2], :].view(1, 1, x.shape[2], -1)
    
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    
    out = torch.stack([out_real, out_imag], dim=-1).flatten(-2)
    return out

# ---------------------------------------------------------------------------
# SOHBET VE ÇIKARIM YÖNETİCİSİ (CHAT MANAGER)
# ---------------------------------------------------------------------------

class StreamedInferenceManager:
    def __init__(
        self,
        model_size_b: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct", # Tokenizer için gerekli
        policy: Optional[CachePolicy] = None,
        gguf_path: Optional[str] = None,
    ):
        self.model_size_b = model_size_b
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        
        print(f">>> [1/2] Tokenizer yükleniyor: {model_id}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Tokenizer yüklenemedi. İnternet bağlantınızı kontrol edin: {e}")

        self.session = LLMTuningTurboSession(
            model_size_b=model_size_b,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            policy=policy,
        )

        self._gguf_path = gguf_path
        self._gguf_map = GGUFMap(gguf_path) if gguf_path else None
        
        # Her katman için sıkıştırılmış KV önbelleğini RAM'de tutacağımız liste
        self.kv_history = [None] * num_layers
        
        # RoPE frekanslarını önceden hesapla
        self.cos_cached, self.sin_cached = precompute_rope_freqs(head_dim, 32768)
        
        # Cache for fixed weights
        self._embd_cached = None
        self._lm_head_cached = None
        
        # Cihaz belirleme (Apple Silicon için MPS kullanmayı dene)
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        
        print(f">>> Motor cihazı: {self.device}")

    @classmethod
    def for_model_size(cls, model_size_b: float, model_id: str = "Qwen/Qwen2.5-32B-Instruct", gguf_path: str = None, policy: Optional[CachePolicy] = None) -> "StreamedInferenceManager":
        configs = {
            8:    dict(num_layers=32, num_heads=32, head_dim=128),
            32:   dict(num_layers=64, num_heads=40, head_dim=128),
            104:  dict(num_layers=96, num_heads=128, head_dim=128),
        }
        nearest = min(configs.keys(), key=lambda k: abs(k - model_size_b))
        cfg = configs[nearest]
        return cls(model_size_b=model_size_b, model_id=model_id, gguf_path=gguf_path, policy=policy, **cfg)

    def _load_layer_weights(self, layer_idx: int) -> dict:
        """Diskten sadece İSTENEN katmanın ağırlıklarını RAM'e çeker."""
        if self._gguf_map:
            return self._gguf_map.get_layer_weights(layer_idx)
        return {} # Fallback

    def _clean_memory(self) -> None:
        """Kullanılmış katman ağırlıklarını RAM'den derhal siler."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # -----------------------------------------------------------------------
    # MATEMATİKSEL İLERİ BESLEME (GERÇEK PYTORCH/NUMPY LAYER CALL BURAYA GELECEK)
    # -----------------------------------------------------------------------
    def _compute_real_layer(self, hidden_state: torch.Tensor, weights: dict, layer_idx: int) -> torch.Tensor:
        """
        Qwen 2.5 Transformer Katman Hesaplaması.
        """
        # 1. Attention Norm
        norm_w = torch.from_numpy(weights["attn_norm_weight"]).to(self.device)
        h = rms_norm(hidden_state, norm_w)
        
        # 2. QKV Projeksiyonları
        # Weights (Q: 5120x5120, K: 1024x5120, V: 1024x5120)
        qw = torch.from_numpy(weights["q_weight"]).to(self.device).T
        kw = torch.from_numpy(weights["k_weight"]).to(self.device).T
        vw = torch.from_numpy(weights["v_weight"]).to(self.device).T
        
        q = (h @ qw).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # GQA: KV heads = 8, Q heads = 40
        num_kv_heads = kw.shape[1] // self.head_dim
        k = (h @ kw).view(1, -1, num_kv_heads, self.head_dim).transpose(1, 2)
        v = (h @ vw).view(1, -1, num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. RoPE Uygulama
        q = apply_rope(q, self.cos_cached.to(self.device), self.sin_cached.to(self.device))
        k = apply_rope(k, self.cos_cached.to(self.device), self.sin_cached.to(self.device))
        
        # 4. KV Cache Yönetimi (TurboQuant)
        if self.kv_history[layer_idx] is not None:
             # Önceki KV'yi aç ve birleştir
             pk, pv = self.session.decompress(self.kv_history[layer_idx])
             past_k = torch.from_numpy(pk).to(self.device).unsqueeze(0)
             past_v = torch.from_numpy(pv).to(self.device).unsqueeze(0)
             k = torch.cat([past_k, k], dim=2)
             v = torch.cat([past_v, v], dim=2)
        
        # Yeni KV'yi sıkıştır ve sakla
        self.kv_history[layer_idx] = self.session.compress(k.squeeze(0).cpu().numpy(), v.squeeze(0).cpu().numpy(), layer_idx)
        
        # 5. Attention Hesaplama (GQA)
        # k: (1, 8, seq, 128) -> repeat to 40 heads
        k_rep = k.repeat_interleave(self.num_heads // num_kv_heads, dim=1)
        v_rep = v.repeat_interleave(self.num_heads // num_kv_heads, dim=1)
        
        # Attention score
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = (q @ k_rep.transpose(-2, -1)) * scale
        
        # Causal mask apply (if seq > 1)
        if attn.shape[-1] > 1:
             mask = torch.triu(torch.ones(attn.shape[-2], attn.shape[-1], device=self.device) * float('-inf'), diagonal=1)
             attn += mask
             
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = (attn @ v_rep).transpose(1, 2).reshape(1, -1, self.d_model)
        
        # 6. Output Projeksiyonu
        ow = torch.from_numpy(weights["o_weight"]).to(self.device).T
        h = hidden_state + (attn_out @ ow)
        
        # 7. FFN (SwiGLU)
        norm_w_ffn = torch.from_numpy(weights["ffn_norm_weight"]).to(self.device)
        h_ffn = rms_norm(h, norm_w_ffn)
        
        gw = torch.from_numpy(weights["ffn_gate_weight"]).to(self.device).T
        uw = torch.from_numpy(weights["ffn_up_weight"]).to(self.device).T
        dw = torch.from_numpy(weights["ffn_down_weight"]).to(self.device).T
        
        # SwiGLU: (silu(gate) * up) * down
        ffn_out = (torch.nn.functional.silu(h_ffn @ gw) * (h_ffn @ uw)) @ dw
        
        return h + ffn_out

    def _lm_head_and_sample(self, hidden_state: torch.Tensor) -> int:
        """Son katmandan çıkan veriyi kelime (token) ID'sine dönüştürür."""
        if self._lm_head_cached is None and self._gguf_map:
            info = self._gguf_map.tensors.get("output.weight")
            if info:
                 start = self._gguf_map.data_start + info["offset"]
                 size_bytes = np.prod(info["shape"]) * 4 # F32 usually
                 data = self._gguf_map.map[start : start + size_bytes]
                 self._lm_head_cached = torch.from_numpy(np.frombuffer(data, dtype=np.float32).reshape(info["shape"])).to(self.device).T
        
        if self._lm_head_cached is not None:
             logits = hidden_state @ self._lm_head_cached
             return int(torch.argmax(logits, dim=-1)[0, -1].item())
        
        return np.random.randint(10, 1000)

    # -----------------------------------------------------------------------
    # GERÇEK ZAMANLI SOHBET VE METİN ÜRETME DÖNGÜSÜ
    # -----------------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 100):
        print(f"\n[Kullanıcı]: {prompt}")
        print(f"[Asistan]: ", end="", flush=True)

        # 1. Metni Tokenlara Çevir
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 2. Embedding (GGUF'tan çek ve cache'le)
        if self._embd_cached is None:
            embd_info = self._gguf_map.tensors.get("token_embd.weight")
            embd_start = self._gguf_map.data_start + embd_info["offset"]
            from turboquant.gguf_reader import dequantize_q4_k
            embd_bytes = (np.prod(embd_info["shape"]) // 256) * 144
            embd_data = self._gguf_map.map[embd_start : embd_start + embd_bytes]
            self._embd_cached = torch.from_numpy(dequantize_q4_k(embd_data, embd_info["shape"])).to(self.device)
        
        current_hidden_state = torch.nn.functional.embedding(input_ids, self._embd_cached)
        
        # 3. PREFILL AŞAMASI
        for layer_idx in range(self.num_layers):
            weights = self._load_layer_weights(layer_idx)
            current_hidden_state = self._compute_real_layer(current_hidden_state, weights, layer_idx)
            del weights
            self._clean_memory()

        # 4. DECODE AŞAMASI
        for step in range(max_new_tokens):
            # Son token'ı al
            last_hidden = current_hidden_state[:, -1:]
            
            # Katmanları sırayla işle
            for layer_idx in range(self.num_layers):
                weights = self._load_layer_weights(layer_idx)
                last_hidden = self._compute_real_layer(last_hidden, weights, layer_idx)
                del weights
                self._clean_memory()
            
            # Sonuçtan kelime ID me
            next_token_id = self._lm_head_and_sample(last_hidden)
            
            word = self.tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Yeni input_ids'i ekle (Embedding)
            next_hidden = torch.nn.functional.embedding(torch.tensor([[next_token_id]], device=self.device), self._embd_cached)
            current_hidden_state = torch.cat([current_hidden_state, next_hidden], dim=1)
                
        print("\n") # Sohbet sonu

    def chat(self):
        """Sürekli çalışan interaktif terminal sohbet arayüzü."""
        print("\n" + "="*50)
        print("TurboQuant+ AirLLM İnteraktif Sohbet Modu")
        print("Çıkmak için 'quit' veya 'exit' yazın.")
        print("="*50)
        
        # Geçmiş hafızayı (KV Cache) sıfırla
        self.kv_history = [None] * self.num_layers
        
        while True:
            user_input = input("\nSen: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input.strip():
                continue
                
            self.generate(user_input, max_new_tokens=50)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="TurboQuant+ AirLLM Streamed Inference (Python Functional Prototype)")
    # llama-cli compatibility flags
    parser.add_argument("-m", "--model", "--gguf-file", dest="gguf_file", help="Path to the GGUF model file")
    parser.add_argument("--model-size", type=float, default=32.0, help="Model size in billions (default: 32.0)")
    parser.add_argument("-n", "--predict", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Inference prompt")
    parser.add_argument("--cache-type-k", type=str, default=None, choices=["turbo2", "turbo3", "turbo4"], 
                       help="TurboQuant K-cache precision (default: auto)")
    parser.add_argument("--cache-type-v", type=str, default=None, choices=["turbo2", "turbo3", "turbo4"], 
                       help="TurboQuant V-cache precision (default: auto)")
    # Dummy flags for compatibility to prevent errors
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=0, help="Layers on GPU (ignored)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="CPU threads (ignored)")
    
    args, unknown = parser.parse_known_args()
    
    if not args.gguf_file:
         parser.print_help()
         sys.exit(1)
    
    # 1. Custom Policy Hazırla
    custom_policy = None
    if args.cache_type_k or args.cache_type_v:
        bits_map = {"turbo2": 2, "turbo3": 3, "turbo4": 4}
        k_bits = bits_map.get(args.cache_type_k, 4)
        v_bits = bits_map.get(args.cache_type_v, 4)
        
        from turboquant.llmtuning_bridge import CachePolicy
        custom_policy = CachePolicy(
            k_bits=k_bits,
            v_bits=v_bits,
            boundary_k_bits=4,
            boundary_n_layers=2,
            max_context=4096
        )
        print("="*60)
        print(">>> [LLMTuning Logic Enabled] <<<")
        print(f">>> Active Flags: K={args.cache_type_k or 'auto'}, V={args.cache_type_v or 'auto'}")
        print(">>> Mode: Memory-Optimized Layer Sharding (AirLLM)")
        print("="*60)

    # 2. Manager'ı oluştur
    from turboquant.llmtuning_bridge import CachePolicy
    manager = StreamedInferenceManager.for_model_size(
        args.model_size, 
        gguf_path=args.gguf_file, 
        policy=custom_policy
    )
    
    # 3. Çalıştır
    if args.prompt:
         manager.generate(args.prompt, max_new_tokens=args.predict)
    else:
         manager.chat()
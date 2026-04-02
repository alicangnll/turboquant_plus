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
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turboquant.llmtuning_bridge import LLMTuningTurboSession, CachePolicy
from turboquant.kv_cache import KVCacheCompressor
from turboquant.gguf_reader import GGUFMap

# ---------------------------------------------------------------------------
# MATH UTILITIES (TORCH ACCELERATED)
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: Root Mean Square Layer Normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

def precompute_rope_freqs(dim: int, end: int, theta: float = 1000000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE (Rotary Positional Embedding) frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """Apply RoPE to Query or Key tensors.
    
    Args:
        x: (batch, heads, seq_len, head_dim)
        cos, sin: precomputed RoPE frequencies
        start_pos: position offset for autoregressive decoding.
                   During prefill start_pos=0; during decode start_pos=current_seq_length.
    """
    seq_len = x.shape[2]
    x_real = x[..., 0::2]
    x_imag = x[..., 1::2]
    
    cos = cos[start_pos : start_pos + seq_len, :].view(1, 1, seq_len, -1)
    sin = sin[start_pos : start_pos + seq_len, :].view(1, 1, seq_len, -1)
    
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
        model_size_b: float = 32.0,
        num_layers: int = 64,
        num_heads: int = 40,
        head_dim: int = 128,
        num_kv_heads: Optional[int] = None,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct",
        policy: Optional[CachePolicy] = None,
        gguf_path: Optional[str] = None,
    ):
        self._gguf_path = gguf_path
        self._gguf_map = GGUFMap(gguf_path) if gguf_path else None
        
        # Default architecture params
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.rope_theta = 1000000.0 if num_layers == 32 else 10000.0
        
        # Override with GGUF Metadata if available
        if self._gguf_map and self._gguf_map.metadata:
            meta = self._gguf_map.metadata
            arch = meta.get("general.architecture", "llama")
            prefix = f"{arch}."
            
            self.num_layers = meta.get(f"{prefix}block_count", meta.get("llama.block_count", self.num_layers))
            self.num_heads = meta.get(f"{prefix}attention.head_count", meta.get("llama.attention.head_count", self.num_heads))
            self.num_kv_heads = meta.get(f"{prefix}attention.head_count_kv", meta.get("llama.attention.head_count_kv", self.num_heads))
            
            embd_len = meta.get(f"{prefix}embedding_length", meta.get("llama.embedding_length", self.num_heads * self.head_dim))
            self.head_dim = embd_len // self.num_heads
            
            self.rope_theta = meta.get(f"{prefix}rope.freq_base", meta.get("llama.rope.freq_base", self.rope_theta))
            
            g_name = str(meta.get("general.name", "")).lower()
            if "llama" in g_name:
                model_id = "meta-llama/Llama-3.1-8B-Instruct"
            elif "qwen" in g_name:
                model_id = "Qwen/Qwen2.5-32B-Instruct"

            print(f">>> [GGUF Metadata Detected]")
            print(f">>> Architecture: {arch} ({g_name})")
            print(f">>> Config: Layers={self.num_layers}, Heads={self.num_heads}, KV-Heads={self.num_kv_heads}, Dim={embd_len}")
            print(f">>> Selected Tokenizer: {model_id}")

        self.model_size_b = model_size_b
        self.d_model = self.num_heads * self.head_dim
        
        print(f">>> [1/2] Tokenizer yükleniyor: {model_id}...")
        self.tokenizer = self._load_tokenizer(model_id)

        self.session = LLMTuningTurboSession(
            model_size_b=model_size_b,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            policy=policy,
        )

        self.kv_history = [None] * self.num_layers
        self._seq_pos = 0  # Tracks the total sequence length for RoPE position offset
        self.cos_cached, self.sin_cached = precompute_rope_freqs(self.head_dim, 32768, theta=self.rope_theta)
        
        self._embd_cached = None
        self._lm_head_cached = None
        self._output_norm_cached = None
        
        # Cihaz ve Hassasiyet Ayarı
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        
        print(f">>> Motor cihazı: {self.device} ({self.dtype})")
        
        # Tüm katman ağırlıklarını RAM'e önbelleğe al (ilk kez = yavaş, sonra = sıfır disk I/O)
        print(f">>> [Perf] Tüm {self.num_layers} katmanın ağırlıkları RAM'e yükleniyor... (bir kez)")
        import time
        t0 = time.time()
        self._weight_cache: dict = {}
        if self._gguf_map:
            for li in range(self.num_layers):
                raw = self._gguf_map.get_layer_weights(li)
                self._weight_cache[li] = {k: torch.from_numpy(v).to(device=self.device, dtype=self.dtype) for k, v in raw.items()}
                if (li + 1) % 8 == 0:
                    print(f">>>   {li+1}/{self.num_layers} katman y\u00fcklendi...")
            
            # Embedding ve LM Head tablosunu da önceden y\u00fckle!
            print(">>>   Embedding tablosu (token_embd) RAM'e y\u00fckleniyor...")
            for candidate in ["token_embd.weight", "tok_embeddings.weight"]:
                t = self._load_any_tensor(candidate)
                if t is not None:
                    self._embd_cached = t
                    break
            
            print(">>>   LM Head (output.weight) RAM'e y\u00fckleniyor...")
            lm_name = "output.weight"
            if lm_name not in self._gguf_map.tensors:
                lm_name = next((c for c in ["token_embd.weight", "tok_embeddings.weight"] if c in self._gguf_map.tensors), None)
            if lm_name:
                t = self._load_any_tensor(lm_name)
                if t is not None:
                    self._lm_head_cached = t.T  # (d_model, vocab_size)

            # Final output norm (output_norm.weight) — critical for correct logits
            print(">>> Output Norm (output_norm.weight) RAM'e yükleniyor...")
            for norm_name in ["output_norm.weight", "norm.weight", "model.norm.weight"]:
                t = self._load_any_tensor(norm_name)
                if t is not None:
                    self._output_norm_cached = t
                    print(f">>>   output_norm yüklendi: {norm_name} shape={t.shape}")
                    break
            if self._output_norm_cached is None:
                print(">>> [WARNING] output_norm.weight bulunamadı! Logitler normalize edilmeyecek.")
        print(f">>> [Perf] T\u00fcm a\u011f\u0131rl\u0131klar haz\u0131r ({time.time()-t0:.1f}s). Art\u0131k s\u0131f\u0131r disk I/O!")

    def _load_tokenizer(self, model_id: str):
        """Load tokenizer with multi-level fallback:
        1. Original HF model_id
        2. Unsloth/community mirror (for gated repos)
        3. Build from GGUF metadata (tokens + merges)
        4. GPT-2 as absolute last resort
        """
        # --- Level 1: Try original model_id ---
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            print(f">>> Tokenizer yüklendi: {model_id}")
            return tok
        except Exception as e:
            print(f">>> [Tokenizer L1] {model_id} erişilemedi: {type(e).__name__}")

        # --- Level 2: Try unsloth / community mirrors ---
        MIRRORS = {
            "meta-llama/Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct": "unsloth/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct": "unsloth/Llama-3.1-70B-Instruct",
        }
        mirror = MIRRORS.get(model_id)
        if mirror:
            try:
                tok = AutoTokenizer.from_pretrained(mirror)
                print(f">>> Tokenizer mirror'dan yüklendi: {mirror}")
                return tok
            except Exception as e:
                print(f">>> [Tokenizer L2] Mirror {mirror} erişilemedi: {type(e).__name__}")

        # --- Level 3: Build from GGUF metadata ---
        if self._gguf_map and self._gguf_map.metadata:
            tok = self._build_tokenizer_from_gguf()
            if tok is not None:
                return tok

        # --- Level 4: Absolute last resort ---
        print(">>> [WARNING] GPT-2 fallback tokenizer kullanılıyor — sonuçlar yanlış olabilir!")
        return AutoTokenizer.from_pretrained("gpt2")

    def _build_tokenizer_from_gguf(self):
        """Build a BPE tokenizer directly from GGUF metadata."""
        meta = self._gguf_map.metadata
        tokens = meta.get("tokenizer.ggml.tokens")
        merges = meta.get("tokenizer.ggml.merges")
        bos_id = meta.get("tokenizer.ggml.bos_token_id", 1)
        eos_id = meta.get("tokenizer.ggml.eos_token_id", 2)

        if not tokens or not merges:
            print(">>> [Tokenizer L3] GGUF tokenizer verisi bulunamadı.")
            return None

        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders
            from tokenizers.processors import TemplateProcessing

            # Build vocabulary: {token_string: id}
            vocab = {tok_str: i for i, tok_str in enumerate(tokens)}

            # Create BPE tokenizer
            bpe = models.BPE(vocab=vocab, merges=[(m.split()[0], m.split()[1]) for m in merges if len(m.split()) == 2])
            tokenizer_obj = Tokenizer(bpe)

            # Llama 3 uses ByteLevel pre-tokenization
            tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer_obj.decoder = decoders.ByteLevel()

            # Wrap in HuggingFace PreTrainedTokenizerFast
            bos_token = tokens[bos_id] if bos_id < len(tokens) else "<s>"
            eos_token = tokens[eos_id] if eos_id < len(tokens) else "</s>"

            tok = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer_obj,
                bos_token=bos_token,
                eos_token=eos_token,
            )

            # Manually ensure eos_token_id matches GGUF
            tok._eos_token_id = eos_id
            tok._bos_token_id = bos_id

            print(f">>> Tokenizer GGUF'tan oluşturuldu (vocab={len(tokens)}, merges={len(merges)}, eos={eos_id})")
            return tok
        except Exception as e:
            print(f">>> [Tokenizer L3] GGUF tokenizer oluşturulamadı: {e}")
            return None

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
        """Önbellekten katman ağırlıklarını döndürür (sıfır disk I/O)."""
        if layer_idx in self._weight_cache:
            return self._weight_cache[layer_idx]
        # Fallback: disk'ten oku (önbellek yoksa)
        if self._gguf_map:
            weights = self._gguf_map.get_layer_weights(layer_idx)
            return {k: torch.from_numpy(v).to(device=self.device, dtype=self.dtype) for k, v in weights.items()}
        return {}

    def _clean_memory(self, force: bool = False) -> None:
        if force:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def _compute_real_layer(self, hidden_state: torch.Tensor, weights: dict, layer_idx: int, start_pos: int = 0) -> torch.Tensor:
        """Run a single transformer layer.
        
        Args:
            hidden_state: (batch, seq_len, d_model)
            weights: layer weight dict
            layer_idx: which layer (for KV cache indexing)
            start_pos: RoPE position offset. 0 during prefill, seq_len during decode.
        """
        # 1. Attn Norm
        h = rms_norm(hidden_state, weights["attn_norm_weight"])
        
        # 2. QKV Proj - Standard GGUF weights are (out, in), so we need .T for (in, out)
        q = (h @ weights["q_weight"].T).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = (h @ weights["k_weight"].T).view(1, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = (h @ weights["v_weight"].T).view(1, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. RoPE — use position offset so decode tokens get correct positions
        cos = self.cos_cached.to(self.device)
        sin = self.sin_cached.to(self.device)
        q = apply_rope(q, cos, sin, start_pos=start_pos).to(self.dtype)
        k = apply_rope(k, cos, sin, start_pos=start_pos).to(self.dtype)
        
        # 4. KV Cache — Raw storage (bypass compression to avoid slogdet instability)
        #    K values are stored POST-RoPE, so no double-application on reload.
        if self.kv_history[layer_idx] is not None:
             past_k, past_v = self.kv_history[layer_idx]
             past_k = torch.from_numpy(past_k).to(device=self.device, dtype=self.dtype).unsqueeze(0)
             past_v = torch.from_numpy(past_v).to(device=self.device, dtype=self.dtype).unsqueeze(0)
             k = torch.cat([past_k, k], dim=2)
             v = torch.cat([past_v, v], dim=2)
        
        # Store raw (n_heads, seq, head_dim) — trim to last 512 tokens to bound memory
        MAX_CTX = 512
        k_np = k.squeeze(0).cpu().to(torch.float16).numpy()[:, -MAX_CTX:, :]
        v_np = v.squeeze(0).cpu().to(torch.float16).numpy()[:, -MAX_CTX:, :]
        self.kv_history[layer_idx] = (k_np, v_np)
        
        # 5. Attention (GQA via SDPA) — manually expand KV heads for MPS
        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            is_causal=(q.shape[2] > 1) 
        ).transpose(1, 2).reshape(1, -1, self.d_model)
        
        # 6. Output
        h = hidden_state + (attn_out @ weights["o_weight"].T)
        
        # 7. FFN
        h_ffn = rms_norm(h, weights["ffn_norm_weight"])
        ffn_out = (torch.nn.functional.silu(h_ffn @ weights["ffn_gate_weight"].T) * (h_ffn @ weights["ffn_up_weight"].T)) @ weights["ffn_down_weight"].T
        
        return h + ffn_out

    def _load_any_tensor(self, name: str) -> Optional[torch.Tensor]:
        """GGUF'taki herhangi bir tensörü tipine göre okur ve döndürür."""
        if name not in self._gguf_map.tensors:
            return None
        info = self._gguf_map.tensors[name]
        start = self._gguf_map.data_start + info["offset"]
        shape = info["shape"]
        t = info["type"]
        n_elements = int(np.prod(shape))

        # GGML Type ID -> bytes per block / elements per block
        # https://github.com/ggerganov/ggml/blob/master/include/ggml.h
        BYTES_PER_ELEM = {
            0: (4, 1),  # F32
            1: (2, 1),  # F16
            2: (18, 32),  # Q4_0
            3: (20, 32),  # Q4_1
            6: (22, 32),  # Q5_0
            7: (24, 32),  # Q5_1
            8: (34, 32),  # Q8_0
            10: (30, 256),  # Q2_K
            11: (52, 256),  # Q3_K
            12: (144, 256), # Q4_K
            13: (176, 256), # Q5_K
            14: (210, 256), # Q6_K
        }

        from turboquant.gguf_reader import dequantize_q4_k, dequantize_q6_k

        if t == 0:  # F32
            data = self._gguf_map.map[start : start + n_elements * 4]
            arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
        elif t == 1:  # F16
            data = self._gguf_map.map[start : start + n_elements * 2]
            arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
        elif t == 12:  # Q4_K
            n_blocks = n_elements // 256
            data = self._gguf_map.map[start : start + n_blocks * 144]
            arr = dequantize_q4_k(data, shape)
        elif t == 14:  # Q6_K
            n_blocks = n_elements // 256
            data = self._gguf_map.map[start : start + n_blocks * 210]
            arr = dequantize_q6_k(data, shape)
        elif t == 8:  # Q8_0
            n_blocks = n_elements // 32
            data = self._gguf_map.map[start : start + n_blocks * 34]
            # Q8_0: 2 bytes scale (f16) + 32 bytes qs
            blocks = data.reshape(n_blocks, 34)
            d = blocks[:, 0:2].view(np.float16).astype(np.float32)
            qs = blocks[:, 2:].astype(np.int8).astype(np.float32)
            arr = (qs * d).reshape(shape).astype(np.float16)
        else:
            # Unknown type: return zeros
            print(f">>> [Warning] Unknown tensor type {t} for '{name}', using zeros.")
            arr = np.zeros(shape, dtype=np.float16)

        return torch.from_numpy(arr.astype(np.float16)).to(device=self.device, dtype=self.dtype)

    def _lm_head_and_sample(self, hidden_state: torch.Tensor) -> int:
        """Apply final RMSNorm + LM head projection, then greedy sample."""
        if self._lm_head_cached is not None:
            h = hidden_state
            # Apply final output norm (RMSNorm) before projection
            if self._output_norm_cached is not None:
                h = rms_norm(h, self._output_norm_cached)
            logits = h @ self._lm_head_cached
            return int(torch.argmax(logits, dim=-1)[0, -1].item())
        return 100

    def generate(self, prompt: str, max_new_tokens: int = 100):
        print(f"\n[Kullanıcı]: {prompt}")
        print(f"[Asistan]: ", end="", flush=True)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        
        if self._embd_cached is None:
            raise RuntimeError("Embedding önbelleği boş!")
            
        # === PREFILL: process all prompt tokens at once ===
        current_hidden_state = torch.nn.functional.embedding(input_ids, self._embd_cached)
        self._seq_pos = 0
        for layer_idx in range(self.num_layers):
            w = self._load_layer_weights(layer_idx)
            current_hidden_state = self._compute_real_layer(current_hidden_state, w, layer_idx, start_pos=0)
            del w
        
        # Sample the first token from the last hidden state of the prompt
        next_token_id = self._lm_head_and_sample(current_hidden_state[:, -1:])
        word = self.tokenizer.decode([next_token_id])
        print(word, end="", flush=True)
        
        self._seq_pos = prompt_len
        self._last_token_id = next_token_id

        # === DECODE: generate subsequent tokens ===
        for step in range(1, max_new_tokens):
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
            # Embed the last generated token
            decode_hidden = torch.nn.functional.embedding(
                torch.tensor([[next_token_id]], device=self.device),
                self._embd_cached
            )
            
            # Run through layers updating KV cache at the current position
            for layer_idx in range(self.num_layers):
                w = self._load_layer_weights(layer_idx)
                decode_hidden = self._compute_real_layer(decode_hidden, w, layer_idx, start_pos=self._seq_pos)
                del w
            
            self._seq_pos += 1
            next_token_id = self._lm_head_and_sample(decode_hidden)
            word = self.tokenizer.decode([next_token_id])
            print(word, end="", flush=True)
            
        print("\n")

    def chat(self):
        print("\n" + "="*50)
        print("TurboQuant+ AirLLM İnteraktif Sohbet Modu")
        print("Çıkmak için 'quit' veya 'exit' yazın.")
        print("="*50)
        self.kv_history = [None] * self.num_layers
        self._seq_pos = 0
        self._last_token_id = 0
        while True:
            user_input = input("\nSen: ")
            if user_input.lower() in ['quit', 'exit']: break
            if not user_input.strip(): continue
            self.generate(user_input, max_new_tokens=50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="gguf_file")
    parser.add_argument("--model-size", type=float, default=32.0)
    parser.add_argument("-n", "--predict", type=int, default=100)
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("--cache-type-k", type=str, default="turbo4")
    parser.add_argument("--cache-type-v", type=str, default="turbo2")
    args, _ = parser.parse_known_args()
    
    if not args.gguf_file: sys.exit(1)
    
    bits_map = {"turbo2": 2, "turbo3": 3, "turbo4": 4}
    policy = CachePolicy(k_bits=bits_map.get(args.cache_type_k, 4), v_bits=bits_map.get(args.cache_type_v, 2), boundary_k_bits=4, boundary_n_layers=2, max_context=4096)
    
    manager = StreamedInferenceManager(model_size_b=args.model_size, gguf_path=args.gguf_file, policy=policy)
    if args.prompt: manager.generate(args.prompt, max_new_tokens=args.predict)
    else: manager.chat()
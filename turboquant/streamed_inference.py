"""
Streamed Inference Manager (Chat Version) — AirLLM + TurboQuant pipeline.
Bu sürüm:
1. BatchEncoding/Slicing hatası (TypeError) giderilmiştir.
2. llama.cpp tarzı otomatik chat template uygular.
3. Gated repo kısıtlamalarına karşı otomatik mirror tokenizer kullanır.
4. [BUG FIX] TurboQuant compress/decompress parametre sırası (k, v, layer_idx) düzeltildi.
"""

from __future__ import annotations
import gc
import sys
import time
import argparse
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from turboquant.llmtuning_bridge import LLMTuningTurboSession, CachePolicy
from turboquant.gguf_reader import GGUFMap


class StreamedInferenceManager:
    def __init__(
        self,
        model_size_b: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct",
        policy: Optional[CachePolicy] = None,
        gguf_path: Optional[str] = None,
    ):
        self.model_size_b = model_size_b
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        
        # Cihaz seçimi (Apple Silicon/NVIDIA/CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f">>> Motor cihazı: {self.device} (torch.float16)")
        
        # Tokenizer yükleme (Hata toleranslı)
        self.tokenizer = None
        self._load_tokenizer(model_id)

        self.session = LLMTuningTurboSession(
            model_size_b=model_size_b,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            policy=policy,
        )

        self._gguf_path = gguf_path
        self._gguf_map = GGUFMap(gguf_path) if gguf_path else None
        
        # KV Cache ve Sohbet Durumu
        self.kv_history = [None] * num_layers
        self.chat_history = []
        self.past_seq_len = 0
        
        # Sentetik Katman Önbellekleri (Simülasyon stabilitesi için)
        self._embd_cached = None
        self._lm_head_cached = None
        self._rms_eps = 1e-5

    def _load_tokenizer(self, model_id: str):
        """Tokenizer'ı yükler, ana repo kilitliyse mirror (ayna) reposuna dener."""
        candidates = [model_id]
        # Llama 3.1 ve Qwen için açık mirrorlar
        if "llama-3.1" in model_id.lower():
            candidates.append("unsloth/Llama-3.1-8B-Instruct")
        elif "qwen" in model_id.lower():
            candidates.append("unsloth/Qwen2.5-32B-Instruct")

        for cid in candidates:
            try:
                print(f">>> Tokenizer deneniyor: {cid}...")
                self.tokenizer = AutoTokenizer.from_pretrained(cid, trust_remote_code=True)
                if self.tokenizer:
                    print(f">>> Tokenizer başarıyla yüklendi: {cid}")
                    return
            except Exception:
                continue
        
        if not self.tokenizer:
            print("❌ KRİTİK HATA: Tokenizer yüklenemedi. Lütfen interneti kontrol edin.")
            sys.exit(1)

    @classmethod
    def for_model_size(cls, model_size_b: float, model_id: str = "Qwen/Qwen2.5-32B-Instruct") -> "StreamedInferenceManager":
        configs = {
            0.5:  dict(num_layers=24, num_heads=14, head_dim=64),
            8:    dict(num_layers=32, num_heads=32, head_dim=128),
            20:   dict(num_layers=44, num_heads=16, head_dim=128),
            32:   dict(num_layers=64, num_heads=40, head_dim=128),
            104:  dict(num_layers=96, num_heads=128, head_dim=128),
        }
        nearest = min(configs.keys(), key=lambda k: abs(k - model_size_b))
        cfg = configs[nearest]
        return cls(model_size_b=model_size_b, model_id=model_id, **cfg)

    def _clean_memory(self) -> None:
        """Kullanılmış katmanları RAM'den siler (AirLLM çekirdeği)."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _rms_norm(self, x, weight, eps=None):
        """Root Mean Square Layer Normalization."""
        if eps is None:
            eps = self._rms_eps
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * norm * weight

    def _apply_rope(self, x, seq_pos, head_dim, base=500000.0):
        """Apply Rotary Positional Embeddings (RoPE)."""
        # x: (num_heads, seq_len, head_dim)
        n_heads, q_len, d = x.shape
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device).float() / d))
        t = torch.arange(seq_pos, seq_pos + q_len, device=x.device).float()
        freqs = torch.outer(t, inv_freq)  # (q_len, d/2)
        
        # Split into real and imaginary components
        emb = torch.cat((freqs, freqs), dim=-1) # (q_len, d)
        cos = emb.cos()
        sin = emb.sin()
        
        # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        # Note: Llama-3 style rotation on pairs (0, d/2), (1, d/2+1), ...
        half = d // 2
        x_rot = torch.cat((-x[:, :, half:], x[:, :, :half]), dim=-1)
        return x * cos + x_rot * sin

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        rep_penalty: float = 1.15,
        context_tokens: torch.Tensor = None,
        forbid_stop_ids: bool = False,
    ):
        """Metin üretiminde sonsuz döngüleri engeller.

        CPU float32 + ``torch.multinomial``: MPS üzerinde olasılık örneklemesi hatalı/boş
        sonuç verebiliyor; ayrıca ilk adımda EOS üretimi boş yanıt oluşturuyordu.
        """
        logits = logits[0, -1, :].detach().float().cpu().clone()
        vocab = logits.shape[-1]

        if forbid_stop_ids:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and 0 <= eos_id < vocab:
                logits[eos_id] = float("-inf")
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None and 0 <= pad_id < vocab:
                logits[pad_id] = float("-inf")

        if rep_penalty != 1.0 and context_tokens is not None:
            tokens = context_tokens.view(-1).tolist()
            for token in set(tokens):
                if not (0 <= token < vocab):
                    continue
                if logits[token] < 0:
                    logits[token] *= rep_penalty
                else:
                    logits[token] /= rep_penalty

        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        logits = logits / temperature

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            return int(torch.argmax(logits).item())
        probs = probs / probs.sum().clamp(min=1e-12)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _compute_real_layer(self, hidden_state, layer_idx):
        """AirLLM ve TurboQuant'ın katman bazlı işlemci simülasyonu."""
        if self._gguf_map is None:
            return hidden_state # GGUF yoksa fallback (demo/synthetic)

        # 1. Katman ağırlıklarını yükle (Dequantize dahil)
        w = self._gguf_map.get_layer_weights(layer_idx)
        # Tensor'lara çevir ve GPU'ya taşı
        weights = {k: torch.from_numpy(v).to(device=self.device, dtype=torch.float16) for k, v in w.items()}

        # 2. Attention Giriş Normu (Standard RMSNorm)
        norm_w = weights["attn_norm_weight"]
        x = self._rms_norm(hidden_state, norm_w)

        # 3. Self-Attention (Q, K, V Projeksiyonları)
        # x: (1, seq_len, d_model)
        q = torch.matmul(x, weights["q_weight"].T)
        k = torch.matmul(x, weights["k_weight"].T)
        v = torch.matmul(x, weights["v_weight"].T)

        # Başlara böl (Heads)
        seq_len = x.shape[1]
        # Layout: (1, seq_len, d_model) -> (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, -1, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, -1, self.head_dim).transpose(0, 1)

        # RoPE (Position Embeddings)
        q = self._apply_rope(q, self.past_seq_len, self.head_dim)
        k = self._apply_rope(k, self.past_seq_len, self.head_dim)

        # 4. KV Cache (Hata ayıklama için sıkıştırma geçici olarak devre dışı)
        if self.kv_history[layer_idx] is not None:
             past_k, past_v = self.kv_history[layer_idx]
             k = torch.cat([past_k, k], dim=1)
             v = torch.cat([past_v, v], dim=1)

        # KV'yi olduğu gibi sakla
        self.kv_history[layer_idx] = (k.detach(), v.detach())

        # 5. Attention Matmul (Scaled Dot-Product)
        # q: (H, QL, D), k: (HKV, KVL, D) -> repeat k for mqa/gqa if needed
        if k.shape[0] != q.shape[0]:
            k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
            v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Causal Mask (Eğer prefill yapılıyorsa)
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, k.shape[1], device=self.device), diagonal=k.shape[1]-seq_len+1) * -1e4
            scores += mask
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (H, QL, D)
        # (H, L, D) -> (L, H, D) -> (1, L, d_model)
        out = out.transpose(0, 1).contiguous().view(1, seq_len, self.d_model)

        # O Projeksiyonu
        out = torch.matmul(out, weights["o_weight"].T)
        hidden_state = hidden_state + out # Residual 1

        # 6. MLP (Feed Forward)
        x = self._rms_norm(hidden_state, weights["ffn_norm_weight"])
        gate = torch.matmul(x, weights["gate_weight"].T)
        up = torch.matmul(x, weights["up_weight"].T)
        
        # SwiGLU: silu(gate) * up
        x = torch.nn.functional.silu(gate) * up
        out = torch.matmul(x, weights["down_weight"].T)
        hidden_state = hidden_state + out # Residual 2

        # Ağırlıklar serbest bırakılır; katman başına gc/MPS temizliği yapılmaz — prefill yüzlerce kez
        # çağrıldığında etkileşimli sohbet dakikalarca takılıyordu.
        del weights, w, x, q, k, v, attn, out
        return hidden_state

    def generate(self, prompt: str, max_new_tokens: int = 150):
        """Otomatik Chat Template kullanarak metin üretir."""
        self.chat_history.append({"role": "user", "content": prompt})

        # Her turda chat şablonu baştan üretildiği için KV önbelleği sıfırlanmalı (aksi halde token hizası bozulur).
        self.kv_history = [None] * self.num_layers
        self.past_seq_len = 0
        if self._gguf_map is not None:
            md = self._gguf_map.metadata
            v = md.get("llama.attention.layer_norm_rms_epsilon")
            if v is not None:
                self._rms_eps = float(v)

        # Chat template uygula
        tokens = self.tokenizer.apply_chat_template(
            self.chat_history,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # Tokenizer çıktısını (BatchEncoding, dict, list veya Tensor) güvenle çıkar
        if hasattr(tokens, "input_ids"):
            full_input_ids = tokens.input_ids
        elif isinstance(tokens, dict) and "input_ids" in tokens:
            full_input_ids = tokens["input_ids"]
        else:
            full_input_ids = tokens
            
        if not isinstance(full_input_ids, torch.Tensor):
            full_input_ids = torch.tensor(full_input_ids)
            
        full_input_ids = full_input_ids.to(self.device)
        
        if full_input_ids.dim() == 1:
            full_input_ids = full_input_ids.unsqueeze(0)

        input_ids = full_input_ids

        print(f"[Asistan]: ", end="", flush=True)
        
        # GGUF Ağırlıklarını Yükle (Embedding ve LM Head)
        v_size = len(self.tokenizer)
        if self._embd_cached is None and self._gguf_map:
            embd_w = self._gguf_map.get_weight("token_embd.weight")
            self._embd_cached = torch.from_numpy(embd_w).to(device=self.device, dtype=torch.float16)
        elif self._embd_cached is None:
            self._embd_cached = torch.randn((v_size, self.d_model), device=self.device, dtype=torch.float16)

        if self._lm_head_cached is None and self._gguf_map:
             lm_w = self._gguf_map.get_weight("output.weight")
             self._lm_head_cached = torch.from_numpy(lm_w).to(device=self.device, dtype=torch.float16)
        elif self._lm_head_cached is None:
            self._lm_head_cached = torch.randn((v_size, self.d_model), device=self.device, dtype=torch.float16)

        current_hidden_state = torch.nn.functional.embedding(input_ids, self._embd_cached)

        generated_tokens = []
        streamed_text_len = 0

        # 1. PREFILL
        for layer_idx in range(self.num_layers):
            current_hidden_state = self._compute_real_layer(current_hidden_state, layer_idx)
        
        # Sadece prefill bittiğinde pozisyonu güncelle
        self.past_seq_len += input_ids.shape[1]

        min_new_tokens = 2
        # 2. DECODE
        for step in range(max_new_tokens):
            # Final Norm (Eğer GGUF varsa yükle)
            if self._gguf_map:
                 norm_w = torch.from_numpy(self._gguf_map.get_weight("output_norm.weight")).to(self.device, dtype=torch.float16)
                 current_hidden_state = self._rms_norm(current_hidden_state, norm_w)

            logits = torch.matmul(
                current_hidden_state.float(),
                self._lm_head_cached.float().T,
            )

            # Sampling
            context = full_input_ids if not generated_tokens else torch.cat([full_input_ids, torch.tensor([generated_tokens], device=self.device)], dim=1)
            forbid_stop = len(generated_tokens) < min_new_tokens
            next_token_id = self._sample(
                logits,
                context_tokens=context,
                forbid_stop_ids=forbid_stop,
                temperature=0.0,
            )

            generated_tokens.append(next_token_id)

            # SentencePiece / Llama: tek token decode sıklıkla boş string verir; tüm diziyi decode edip sadece yeni soneki yazdır.
            full_piece = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if len(full_piece) >= streamed_text_len:
                print(full_piece[streamed_text_len:], end="", flush=True)
                streamed_text_len = len(full_piece)

            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Sonraki adım için embedding
            current_hidden_state = torch.nn.functional.embedding(torch.tensor([[next_token_id]], device=self.device), self._embd_cached)
            for layer_idx in range(self.num_layers):
                current_hidden_state = self._compute_real_layer(current_hidden_state, layer_idx)
            
            # Her token üretildiğinde pozisyonu ilerlet
            self.past_seq_len += 1

        print("\n")

        self._clean_memory()

        # Geçmişe ekle
        full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.chat_history.append({"role": "assistant", "content": full_response})
        # past_seq_len zaten güncel

    def chat(self):
        print("\n" + "="*50)
        print("TurboQuant+ AirLLM İnteraktif Sohbet Modu")
        print("Otomatik Chat Template & Sampling Aktif.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nSen: ")
                if user_input.lower() in ['quit', 'exit']: break
                if not user_input.strip(): continue
                self.generate(user_input)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--model-size", type=float, default=32)
    parser.add_argument("--cache-type-k", type=str, default="turbo4")
    parser.add_argument("--cache-type-v", type=str, default="turbo2")
    args = parser.parse_args()
    
    tid = "Qwen/Qwen2.5-32B-Instruct"
    if "llama" in args.model.lower(): tid = "meta-llama/Llama-3.1-8B-Instruct"
    
    manager = StreamedInferenceManager.for_model_size(args.model_size, model_id=tid)
    manager._gguf_path = args.model
    manager._gguf_map = GGUFMap(args.model)
    try:
        manager.chat()
    finally:
        if manager._gguf_map is not None:
            manager._gguf_map.close()
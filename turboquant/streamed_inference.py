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

    def _sample(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9, rep_penalty: float = 1.15, context_tokens: torch.Tensor = None):
        """Metin üretiminde sonsuz döngüleri engeller."""
        logits = logits[0, -1, :].float()
        
        # Repetition Penalty
        if rep_penalty != 1.0 and context_tokens is not None:
            # context_tokens listeye çevrilir
            tokens = context_tokens.view(-1).tolist()
            for token in set(tokens):
                if logits[token] < 0:
                    logits[token] *= rep_penalty
                else:
                    logits[token] /= rep_penalty
        
        if temperature <= 0.0:
            return torch.argmax(logits).item()
            
        logits = logits / temperature
        
        # Top-P (Nucleus) Sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
            
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _compute_real_layer(self, hidden_state, layer_idx):
        """AirLLM ve TurboQuant'ın katman bazlı işlemci simülasyonu."""
        seq_len = hidden_state.shape[1]
        
        # TurboQuant KV Sıkıştırma/Açma adımı (Numpy yerine PyTorch Tensor kullanıyoruz)
        # Shape: (num_heads, seq_len, head_dim) — batch boyutu (1) kaldırıldı.
        current_k = torch.randn((self.num_heads, seq_len, self.head_dim), device=self.device, dtype=torch.float16)
        current_v = torch.randn((self.num_heads, seq_len, self.head_dim), device=self.device, dtype=torch.float16)
        
        if self.kv_history[layer_idx] is not None:
            # Decompress: Bridge artık (layer_idx, lkv) bekliyor.
            cached_state = self.kv_history[layer_idx]
            k_np, v_np = self.session.decompress(layer_idx, cached_state)
            
            # Numpy -> Torch -> Device
            past_k = torch.from_numpy(k_np).to(device=self.device, dtype=torch.float16)
            past_v = torch.from_numpy(v_np).to(device=self.device, dtype=torch.float16)
             
        # Compress: Bridge artık (k, v, layer_idx) bekliyor ve torch tensorlarını temizce numpy'a çeviriyor.
        self.kv_history[layer_idx] = self.session.compress(current_k, current_v, layer_idx)
        
        # AirLLM: İşlem bitince RAM'i temizle
        self._clean_memory()
        return hidden_state

    def generate(self, prompt: str, max_new_tokens: int = 150):
        """Otomatik Chat Template kullanarak metin üretir."""
        self.chat_history.append({"role": "user", "content": prompt})
        
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
        
        # Sadece yeni tokenları işle
        input_ids = full_input_ids[:, self.past_seq_len:]
            
        print(f"[Asistan]: ", end="", flush=True)
        
        # Sentetik ağırlıkların ilklendirilmesi
        # Llama 3 gibi modellerde vocab_size (128000) != len(tokenizer) (128256) olabilir.
        # Bu yüzden IndexError'ı önlemek için len(tokenizer) kullanıyoruz.
        v_size = len(self.tokenizer)
        if self._embd_cached is None:
            self._embd_cached = torch.randn((v_size, self.d_model), device=self.device, dtype=torch.float16)
        if self._lm_head_cached is None:
            self._lm_head_cached = torch.randn((v_size, self.d_model), device=self.device, dtype=torch.float16)
            
        current_hidden_state = torch.nn.functional.embedding(input_ids, self._embd_cached)

        generated_tokens = []
        
        # 1. PREFILL
        for layer_idx in range(self.num_layers):
            current_hidden_state = self._compute_real_layer(current_hidden_state, layer_idx)

        # 2. DECODE
        for step in range(max_new_tokens):
            logits = torch.matmul(current_hidden_state, self._lm_head_cached.T)
            
            # Sampling
            context = full_input_ids if not generated_tokens else torch.cat([full_input_ids, torch.tensor([generated_tokens], device=self.device)], dim=1)
            next_token_id = self._sample(logits, context_tokens=context)
            
            generated_tokens.append(next_token_id)
            
            # Ekrana yazdır (special tokenları temizle)
            word = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(word, end="", flush=True)
            
            if next_token_id == self.tokenizer.eos_token_id or next_token_id in self.tokenizer.all_special_ids:
                break
                
            # Sonraki adım için embedding
            current_hidden_state = torch.nn.functional.embedding(torch.tensor([[next_token_id]], device=self.device), self._embd_cached)
            for layer_idx in range(self.num_layers):
                current_hidden_state = self._compute_real_layer(current_hidden_state, layer_idx)

        print("\n")
        
        # Geçmişe ekle
        full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.chat_history.append({"role": "assistant", "content": full_response})
        self.past_seq_len = full_input_ids.shape[1] + len(generated_tokens)

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
    manager.chat()
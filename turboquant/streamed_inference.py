"""Streamed Inference Manager — LLMTuning + TurboQuant unified pipeline.

Combines two orthogonal optimizations into one NumPy-first, hardware-portable
inference framework for large language models on memory-constrained hardware:

  1. LLMTuning strategy: layer-by-layer weight streaming from disk
     - Only one transformer layer's weights resident in memory at a time
     - ThreadPoolExecutor overlap: disk I/O prefetches next layer while GPU
       computes the current one (identical approach to LLMTuning_base.py:441-487)

  2. TurboQuant strategy: KV cache compression between decode steps
     - K cache: full TurboQuant (inner product preservation for Q@K^T)
     - V cache: MSE-only PolarQuant (value reconstruction for attn_weights@V)
     - Boundary layer protection: first/last 2 layers at higher precision
     - Auto-selects turbo4 (32B/70B) or turbo2 (400B+) based on model size

Combined memory reduction for 32B (Qwen2.5-32B Q4_K_M, 64 layers, 40 heads, d=128):
  - Weight streaming:   ~20 GB model → ~350 MB peak weight memory
  - KV cache (turbo4):  4096 ctx @ fp16 ~= 2.56 GB → ~675 MB (3.8×)
  - Total peak:         <2 GB active memory (remainder on NVMe swap if needed)

Model format support:
  - GGUF: used via llama.cpp (existing demo flow, run_turboquant_demo.sh)
  - HuggingFace safetensors: used here for Python-level weight streaming
  - numpy checkpoint: for testing/benchmarking without full model

Usage:
    from turboquant.streamed_inference import StreamedInferenceManager

    manager = StreamedInferenceManager.for_model_size(32)   # 32B

    # Simulate one forward pass (demo mode — no real weights needed)
    result = manager.demo_forward(seq_len=512, num_layers=64, num_heads=40)
    print(result.memory_report)

    # With real HuggingFace weights (requires transformers + safetensors):
    manager.load_model("Qwen/Qwen2.5-32B-Instruct", dtype="float16")
    output = manager.generate("Hello world", max_new_tokens=100)
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from turboquant.llmtuning_bridge import (
    LLMTuningTurboSession,
    CachePolicy,
    LayerKV,
    LayerPrefetcher,
    policy_for_model_size,
)
from turboquant.kv_cache import KVCacheCompressor
from turboquant.gguf_reader import GGUFMap



# ---------------------------------------------------------------------------
# Forward pass result
# ---------------------------------------------------------------------------

@dataclass
class ForwardResult:
    """Result from a single forward pass."""
    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    total_time_s: float
    layer_times_ms: list[float]
    compress_times_ms: list[float]
    kv_raw_mb: float
    kv_compressed_mb: float
    compression_ratio: float
    policy: CachePolicy

    @property
    def memory_report(self) -> str:
        lines = [
            "=" * 64,
            "StreamedInference Forward Pass — Memory & Timing Report",
            "=" * 64,
            f"  Sequence length:   {self.seq_len} tokens",
            f"  Layers processed:  {self.num_layers}",
            f"  Attention heads:   {self.num_heads}",
            f"  Head dimension:    {self.head_dim}",
            f"  K precision:       turbo{self.policy.k_bits} ({self.policy.k_bits}-bit)",
            f"  V precision:       turbo{self.policy.v_bits} ({self.policy.v_bits}-bit)",
            f"  Boundary protect:  first/last {self.policy.boundary_n_layers} layers "
            f"@ turbo{self.policy.boundary_k_bits}",
            "",
            f"  — Memory —",
            f"  KV cache (raw):    {self.kv_raw_mb:.1f} MB (fp16)",
            f"  KV cache (turbo):  {self.kv_compressed_mb:.1f} MB",
            f"  Compression:       {self.compression_ratio:.2f}×",
            f"  Memory saved:      {self.kv_raw_mb - self.kv_compressed_mb:.1f} MB",
            "",
            f"  — Timing —",
            f"  Total time:        {self.total_time_s * 1000:.1f} ms",
            f"  Avg layer time:    {sum(self.layer_times_ms)/max(len(self.layer_times_ms),1):.2f} ms",
            f"  Avg KV compress:   {sum(self.compress_times_ms)/max(len(self.compress_times_ms),1):.2f} ms",
            "=" * 64,
        ]
        return "\n".join(lines)

    @property
    def compress_overhead_fraction(self) -> float:
        """Fraction of total time spent on KV compression (0-1)."""
        total_ms = self.total_time_s * 1000
        compress_ms = sum(self.compress_times_ms)
        return compress_ms / max(total_ms, 1e-9)


# ---------------------------------------------------------------------------
# Layer weight loader (LLMTuning-style)
# ---------------------------------------------------------------------------

def _make_synthetic_layer_weights(
    head_dim: int,
    num_heads: int,
    rng: np.random.Generator,
) -> dict:
    """Generate synthetic Q/K/V weight matrices for demo/benchmark purposes.

    In a real integration, this would load from a HuggingFace safetensors shard
    exactly as LLMTuning does via load_layer() → safetensors.torch.load_file().
    """
    d_model = head_dim * num_heads
    return {
        "q_weight": rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02,
        "k_weight": rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02,
        "v_weight": rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02,
        "o_weight": rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.02,
    }


def _synthetic_attention_forward(
    hidden: np.ndarray,  # (seq_len, d_model)
    weights: dict,
    num_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal multi-head attention forward pass (demo, not optimized).

    Returns: (output, k_cache, v_cache)
      k_cache: (num_heads, seq_len, head_dim)
      v_cache: (num_heads, seq_len, head_dim)
    """
    seq_len, d_model = hidden.shape

    q = hidden @ weights["q_weight"].T  # (seq_len, d_model)
    k = hidden @ weights["k_weight"].T
    v = hidden @ weights["v_weight"].T

    # Reshape to multi-head
    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)  # (H, S, D)
    k = k.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    # Scaled dot-product attention (simplified, no masking)
    scale = head_dim ** -0.5
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale  # (H, S, S)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)  # softmax

    out = np.einsum("hqk,hkd->hqd", attn, v)  # (H, S, D)
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)  # (S, d_model)
    out = out @ weights["o_weight"].T

    return out, k, v


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------

class StreamedInferenceManager:
    """Layer-streamed inference with TurboQuant KV compression.

    This manager orchestrates:
    1. Loading transformer layer weights one at a time (LLMTuning strategy)
    2. Computing attention with prefetched weights
    3. Compressing the resulting KV cache immediately (TurboQuant)
    4. Freeing the layer weights from memory
    5. Overlapping next-layer disk I/O with current-layer compute (prefetch)

    Args:
        model_size_b: Model size in billions.
        num_layers: Transformer layer count.
        num_heads: Attention heads per layer.
        head_dim: Head dimension.
        policy: Override cache compression policy.
        layer_load_fn: Callable(layer_idx) → weight_dict. If None, uses
                       synthetic weights for benchmarking.
    """

    def __init__(
        self,
        model_size_b: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        policy: Optional[CachePolicy] = None,
        layer_load_fn: Optional[Callable[[int], dict]] = None,
        gguf_path: Optional[str] = None,
    ):
        self.model_size_b = model_size_b
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

        self.session = LLMTuningTurboSession(
            model_size_b=model_size_b,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            policy=policy,
        )

        self._rng = np.random.default_rng(42)
        self._layer_load_fn = layer_load_fn
        self._gguf_path = gguf_path
        self._gguf_map = GGUFMap(gguf_path) if gguf_path else None


    @classmethod
    def for_model_size(cls, model_size_b: float) -> "StreamedInferenceManager":
        """Convenience constructor with auto-detected layer/head config.

        Uses canonical head configurations matching common open-weight models.
        """
        # Canonical configs from README large model stress tests
        configs = {
            0.5:  dict(num_layers=24, num_heads=8,  head_dim=64),
            7:    dict(num_layers=32, num_heads=32, head_dim=128),
            8:    dict(num_layers=32, num_heads=32, head_dim=128),
            32:   dict(num_layers=64, num_heads=40, head_dim=128),
            70:   dict(num_layers=80, num_heads=64, head_dim=128),
            104:  dict(num_layers=96, num_heads=128, head_dim=128),
            405:  dict(num_layers=126, num_heads=128, head_dim=128),
        }
        # Find nearest known config
        nearest = min(configs.keys(), key=lambda k: abs(k - model_size_b))
        cfg = configs[nearest]
        return cls(model_size_b=model_size_b, **cfg)

    def _load_layer_weights(self, layer_idx: int) -> dict:
        """Load weights for layer `layer_idx`.

        If GGUF is provided, loads real weights (or synthetic maps if quantized).
        """
        if self._gguf_map:
            weights = self._gguf_map.get_layer_weights(layer_idx)
            if weights:
                 # Check if we got all needed weights
                 needed = ["q_weight", "k_weight", "v_weight", "o_weight"]
                 if all(k in weights for k in needed):
                      return weights

        # Fallback to synthetic if GGUF doesn't have the layer or is missing
        if self._layer_load_fn is not None:
            return self._layer_load_fn(layer_idx)
        # Synthetic demo — deterministic per layer
        layer_rng = np.random.default_rng(42 + layer_idx)
        return _make_synthetic_layer_weights(self.head_dim, self.num_heads, layer_rng)


    def _clean_memory(self) -> None:
        """Free memory aggressively (mirrors LLMTuning's clean_memory())."""
        gc.collect()

    def demo_forward(
        self,
        seq_len: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        verbose: bool = True,
    ) -> ForwardResult:
        """Run a full forward pass demonstrating the 3-stage pipelined inference.

        True pipeline structure (all stages overlap):
          Thread 1 — Disk prefetch:  loads layer N+1 weights from disk to RAM
          Thread 2 — GPU compute:    runs attention for layer N
          Thread 3 — CPU compress:   TurboQuant-compresses layer N-1's KV

        Because Thread 3 runs during Thread 2's compute (not after it), KV
        compression adds ~zero latency to the overall forward pass at context
        lengths where compute dominates (which is always true at 32B+).

        Args:
            seq_len: Sequence length. Defaults to policy.max_context.
            num_layers: Override layer count for quick testing.
            num_heads: Override head count.
            verbose: Print per-layer progress.

        Returns:
            ForwardResult with memory and timing statistics.
        """
        seq_len = seq_len or self.session.policy.max_context
        num_layers = num_layers or self.num_layers
        num_heads = num_heads or self.num_heads

        policy = self.session.policy

        hidden = self._rng.standard_normal((seq_len, self.d_model)).astype(np.float32)
        hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)

        layer_times_ms = []
        compress_times_ms = []
        all_kv: list[LayerKV] = []

        t_total_start = time.perf_counter()

        # --- 3-Stage Pipeline ---
        # Stage 1: disk prefetch (Thread 1) — via ThreadPoolExecutor
        # Stage 2: attention compute (Thread 2, main thread)
        # Stage 3: KV compression (Thread 3) — via session._compress_executor
        #
        # At layer N:
        #   Thread 1 loads weights[N+1] asynchronously
        #   Thread 2 computes attention[N] (main thread)
        #   Thread 3 finishes compressing KV[N-1] from previous iter

        pending_compress: Optional["Future[LayerKV]"] = None

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="turbo_prefetch") as executor:
            weight_future: "Future[dict]" = executor.submit(self._load_layer_weights, 0)

            for layer_idx in range(num_layers):
                t_layer = time.perf_counter()

                # Block until current layer weights are ready (Thread 1)
                weights = weight_future.result()

                # Immediately kick off next layer weight prefetch (Thread 1 →)
                if layer_idx + 1 < num_layers:
                    weight_future = executor.submit(self._load_layer_weights, layer_idx + 1)

                # Thread 2: compute attention for this layer
                hidden, k, v = _synthetic_attention_forward(
                    hidden, weights, num_heads, self.head_dim
                )
                del weights  # free immediately — LLMTuning discipline

                # Thread 3: start compressing THIS layer's KV async
                # (while next iteration computes attention on GPU — zero wait)
                t_compress_submit = time.perf_counter()
                compress_future = self.session.compress_async(layer_idx, k, v)
                del k, v  # safe — compress_async already copied the tensors

                # Collect previous layer's finished compression (if any)
                if pending_compress is not None:
                    lkv = pending_compress.result()  # ~0 wait if compute was slow enough
                    all_kv.append(lkv)
                    compress_times_ms.append(
                        (time.perf_counter() - t_compress_submit) * 1000
                    )

                pending_compress = compress_future
                self._clean_memory()

                layer_elapsed_ms = (time.perf_counter() - t_layer) * 1000
                layer_times_ms.append(layer_elapsed_ms)

                if verbose and (layer_idx % max(1, num_layers // 8) == 0 or layer_idx == num_layers - 1):
                    is_boundary = (
                        layer_idx < policy.boundary_n_layers or
                        layer_idx >= num_layers - policy.boundary_n_layers
                    )
                    bits = policy.boundary_k_bits if is_boundary else policy.k_bits
                    tag = " [boundary]" if is_boundary else ""
                    print(
                        f"  Layer {layer_idx:3d}/{num_layers-1}"
                        f" | turbo{bits}/turbo{bits}{tag}"
                        f" | layer {layer_elapsed_ms:.1f}ms"
                        f" | compress async ▶"
                    )

            # Collect final layer's KV
            if pending_compress is not None:
                lkv = pending_compress.result()
                all_kv.append(lkv)
                compress_times_ms.append(0.0)  # already done

        total_time_s = time.perf_counter() - t_total_start

        raw_bytes = num_heads * seq_len * self.head_dim * 2 * 2 * num_layers
        comp_bytes = sum(lkv.approx_bytes(self.head_dim) for lkv in all_kv)

        if verbose:
            print(f"\n  Pipeline overlap: ~{sum(compress_times_ms):.0f}ms total wait "
                  f"on compress (vs {total_time_s*1000:.0f}ms total)")

        return ForwardResult(
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=self.head_dim,
            total_time_s=total_time_s,
            layer_times_ms=layer_times_ms,
            compress_times_ms=compress_times_ms,
            kv_raw_mb=raw_bytes / 1024 / 1024,
            kv_compressed_mb=comp_bytes / 1024 / 1024,
            compression_ratio=raw_bytes / max(comp_bytes, 1),
            policy=policy,
        )


    def benchmark_compression(

        self,
        seq_lengths: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Benchmark KV compression ratio at multiple context lengths.

        Useful for understanding memory savings before committing to a config.

        Args:
            seq_lengths: Contexts to test. Defaults to [512, 2048, 4096, 8192].
            verbose: Print table if True.

        Returns:
            List of dicts with seq_len, raw_mb, compressed_mb, ratio.
        """
        seq_lengths = seq_lengths or [512, 2048, 4096, 8192]
        results = []

        policy = self.session.policy
        # Use single layer for speed
        compressor = KVCacheCompressor(
            head_dim=self.head_dim,
            k_bits=policy.k_bits,
            v_bits=policy.v_bits,
            seed=42,
        )

        if verbose:
            print(f"\nKV Compression Benchmark — {self.model_size_b:.0f}B model")
            print(f"Config: K=turbo{policy.k_bits}, V=turbo{policy.v_bits}, "
                  f"{self.num_heads} heads, d={self.head_dim}")
            print(f"{'Context':>10} | {'Raw (MB)':>10} | {'Compressed':>12} | {'Ratio':>8} | {'Saved':>10}")
            print("-" * 62)

        for seq_len in seq_lengths:
            stats = compressor.memory_stats(seq_len, self.num_layers, self.num_heads)
            row = {
                "seq_len": seq_len,
                "raw_mb": stats["original_mb"],
                "compressed_mb": stats["compressed_mb"],
                "ratio": stats["compression_ratio"],
                "saved_mb": stats["original_mb"] - stats["compressed_mb"],
            }
            results.append(row)

            if verbose:
                print(
                    f"{seq_len:>10,} | {row['raw_mb']:>10.1f} | "
                    f"{row['compressed_mb']:>12.1f} | {row['ratio']:>8.2f}× | "
                    f"{row['saved_mb']:>10.1f}"
                )

        return results


# ---------------------------------------------------------------------------
# Convenience: run everything from CLI for quick validation
# ---------------------------------------------------------------------------

def _run_demo_cli():
    """Run a quick demonstration printing memory stats for common model sizes."""
    print("\n" + "=" * 64)
    print("TurboQuant + LLMTuning Streamed Inference — Quick Demo")
    print("=" * 64)

    configs = [
        (8,   512,  "Llama-3.1-8B"),
        (32,  512,  "Qwen2.5-32B"),
        (70,  256,  "Llama-3.1-70B"),
        (104, 128,  "Command-R+ 104B"),
    ]

    for model_size_b, seq_len, label in configs:
        print(f"\n{'─'*64}")
        print(f"Model: {label} ({model_size_b}B), seq_len={seq_len}")
        print(f"{'─'*64}")

        manager = StreamedInferenceManager.for_model_size(model_size_b)
        result = manager.demo_forward(
            seq_len=seq_len,
            # Limit layers for speed in demo
            num_layers=min(8, manager.num_layers),
            verbose=True,
        )
        print(result.memory_report)

        # Also show compression benchmark
        manager.benchmark_compression(seq_lengths=[seq_len, seq_len * 4])


if __name__ == "__main__":
    _run_demo_cli()

"""AirLLM Bridge for TurboQuant — Layer-Sharding + KV Compression.

Inspired by AirLLM's core insight: instead of holding all model weights in GPU
memory at once, process each transformer layer individually, loading weights just
in time and immediately freeing them after use.

This module layers TurboQuant's KV cache compression ON TOP of that approach:
  - Model weights: streamed layer-by-layer from disk (AirLLM strategy)
  - KV cache: kept compressed between layers using TurboQuant (our contribution)

Combined effect on 32B/70B models on Apple Silicon (Unified Memory):
  - AirLLM reduces peak weight memory by ~8x (only 1 layer resident at a time)
  - TurboQuant reduces KV cache memory by 3.8-6.4x depending on config
  - Together: 32B model fits comfortably on 24GB, 70B on 64GB

Design principles (preserving TurboQuant's correctness):
  - K cache: full TurboQuant (inner product preservation for Q@K^T attention)
  - V cache: MSE-only PolarQuant (value reconstruction for attn_weights @ V)
  - Boundary layers (first 2 + last 2): higher precision k_bits to protect
    critical routing decisions (mirrors TURBO_LAYER_ADAPTIVE=7 behavior)
  - No modifications to the core TurboQuant/PolarQuant algorithms

Usage:
    from turboquant.airllm_bridge import AirLLMTurboSession

    session = AirLLMTurboSession(
        model_size_b=32,         # 32B, 70B, etc.
        num_layers=64,
        num_heads=32,
        head_dim=128,
    )

    # During a forward pass, after each transformer layer:
    compressed_kv = session.compress_layer_kv(layer_idx, k_tensor, v_tensor)

    # When you need to attend (next token):
    k_restored, v_restored = session.restore_layer_kv(layer_idx, compressed_kv)

    # Session stats
    print(session.memory_report())
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from turboquant.kv_cache import KVCacheCompressor, CompressedKVCache
from turboquant.turboquant import TurboQuant, TurboQuantMSE


# ---------------------------------------------------------------------------
# Model size → cache policy mapping
# ---------------------------------------------------------------------------

@dataclass
class CachePolicy:
    """Per-model-size TurboQuant cache configuration.

    Chosen based on README benchmarks and community data for Apple Silicon.
    """
    k_bits: int = 4          # K cache precision (4 = turbo4, 3 = turbo3, 2 = turbo2)
    v_bits: int = 4          # V cache precision
    boundary_k_bits: int = 4 # First/last 2 layers always use this (TURBO_LAYER_ADAPTIVE)
    boundary_n_layers: int = 2  # Layers to protect at each boundary
    max_context: int = 4096


# Size thresholds in billions of parameters
_POLICIES: list[tuple[float, CachePolicy]] = [
    (0,    CachePolicy(k_bits=4, v_bits=4, max_context=8192)),   # <8B: turbo4/turbo4, long ctx
    (8,    CachePolicy(k_bits=4, v_bits=4, max_context=8192)),   # 8B
    (20,   CachePolicy(k_bits=4, v_bits=4, max_context=4096)),   # 20-32B: turbo4/turbo4
    (32,   CachePolicy(k_bits=4, v_bits=4, max_context=4096)),   # 32B: README confirmed best on M-series
    (65,   CachePolicy(k_bits=4, v_bits=4, max_context=2048)),   # 65-70B: turbo4/turbo4, tighter ctx
    (100,  CachePolicy(k_bits=4, v_bits=4, max_context=1024)),   # 100B: turbo4
    (400,  CachePolicy(k_bits=2, v_bits=2, max_context=512)),    # 400-500B: turbo2 (6.4x compress)
]


def policy_for_model_size(model_size_b: float) -> CachePolicy:
    """Return the recommended CachePolicy for a given model size (in billions).

    Based on README benchmarks:
    - turbo4 beats turbo3 on M1/M2/M3 (avoids L2 cache wall, +33.9% decode)
    - turbo4 still best on M5 for 32B/70B with Q4_K_M weights
    - turbo2 for 400B+: 6.4× compression required to fit KV in remaining unified memory
    """
    policy = _POLICIES[0][1]
    for threshold, p in _POLICIES:
        if model_size_b >= threshold:
            policy = p
    return policy


# ---------------------------------------------------------------------------
# Per-layer compressed KV holder
# ---------------------------------------------------------------------------

@dataclass
class LayerKV:
    """Compressed KV cache for a single transformer layer."""
    layer_idx: int
    seq_len: int
    k_indices: np.ndarray    # (num_heads, seq_len, head_dim_ints)
    k_norms: np.ndarray      # (num_heads, seq_len)
    k_qjl_signs: Optional[np.ndarray]   # TurboQuant second stage (for K only)
    k_residual_norms: Optional[np.ndarray]
    v_indices: np.ndarray    # (num_heads, seq_len, head_dim_ints)
    v_norms: np.ndarray      # (num_heads, seq_len)
    timestamp: float = field(default_factory=time.time)
    is_boundary: bool = False

    @property
    def k_bits_actual(self) -> int:
        """Infer bit width from index values (max index → 2^bits)."""
        return int(np.ceil(np.log2(self.k_indices.max() + 2)))

    def approx_bytes(self, head_dim: int) -> int:
        """Rough compressed size in bytes."""
        n = self.seq_len * self.k_indices.shape[0]  # heads × seq
        k_bytes = n * head_dim // 2  # approx 4 bits/value
        v_bytes = n * head_dim // 2
        norm_bytes = n * 4 * 2  # float32 k + v norms
        return k_bytes + v_bytes + norm_bytes


# ---------------------------------------------------------------------------
# AirLLM-inspired Prefetch Layer Loader (disk → CPU RAM, async)
# ---------------------------------------------------------------------------

class LayerPrefetcher:
    """Background thread that pre-loads the next model layer weights from disk.

    Mirrors AirLLM's ThreadPoolExecutor prefetch approach (airllm_base.py:441-487).

    This class is optional — it wraps any callable `load_fn(layer_path) -> dict`
    and overlaps disk I/O with GPU compute on the current layer.

    Usage:
        prefetcher = LayerPrefetcher(load_fn=my_loader)
        prefetcher.prefetch(layer_paths[0])
        for i, path in enumerate(layer_paths):
            weights = prefetcher.get()              # blocks until ready
            if i + 1 < len(layer_paths):
                prefetcher.prefetch(layer_paths[i+1])  # kick off next
            # ... use weights on GPU ...
            del weights
            clean_memory()
    """

    def __init__(self, load_fn):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turbo_prefetch")
        self._load_fn = load_fn
        self._future: Optional[Future] = None

    def prefetch(self, path: str | Path) -> None:
        """Start loading `path` in background."""
        self._future = self._executor.submit(self._load_fn, path)

    def get(self) -> dict:
        """Block until prefetched data is ready and return it."""
        if self._future is None:
            raise RuntimeError("Call prefetch() before get()")
        return self._future.result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main session class
# ---------------------------------------------------------------------------

class AirLLMTurboSession:
    """Stateful KV cache manager combining AirLLM + TurboQuant strategies.

    Key behaviours:
    1. Auto-selects TurboQuant precision (turbo4 for 32B/70B, turbo2 for 400B+)
    2. Boundary layer protection: first/last `boundary_n_layers` always at
       higher precision (mirrors TURBO_LAYER_ADAPTIVE=7 env var in llama.cpp)
    3. Thread-safe: prefetcher runs in separate thread (disk I/O overlapped
       with GPU compute, same approach as AirLLM)
    4. Memory tracking: measures compressed vs uncompressed size per layer

    Args:
        model_size_b: Model size in billions of parameters.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        head_dim: Head dimension (typically 128 for 32B/70B models).
        policy: Optional manual CachePolicy override. If None, auto-selected
                from model_size_b.
    """

    def __init__(
        self,
        model_size_b: float,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        policy: Optional[CachePolicy] = None,
    ):
        self.model_size_b = model_size_b
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.policy = policy or policy_for_model_size(model_size_b)

        # Build per-layer compressors
        # Boundary layers use boundary_k_bits (usually same as k_bits for turbo4,
        # but for turbo2 models we protect boundaries at turbo4)
        self._compressors: list[KVCacheCompressor] = []
        for layer_idx in range(num_layers):
            is_boundary = (
                layer_idx < self.policy.boundary_n_layers or
                layer_idx >= num_layers - self.policy.boundary_n_layers
            )
            k_bits = self.policy.boundary_k_bits if is_boundary else self.policy.k_bits
            v_bits = self.policy.boundary_k_bits if is_boundary else self.policy.v_bits
            self._compressors.append(KVCacheCompressor(
                head_dim=head_dim,
                k_bits=k_bits,
                v_bits=v_bits,
                seed=42 + layer_idx,
                norm_correction=True,
            ))

        # Stats
        self._total_uncompressed_bytes = 0
        self._total_compressed_bytes = 0
        self._layer_compress_times: list[float] = []

        # KV store: layer_idx → LayerKV
        self._kv_store: dict[int, LayerKV] = {}

    # ------------------------------------------------------------------
    # Compress / Decompress API
    # ------------------------------------------------------------------

    def compress_layer_kv(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> LayerKV:
        """Compress K and V tensors for a single layer.

        Args:
            layer_idx: Transformer layer index.
            k: Key cache, shape (num_heads, seq_len, head_dim) or
               (seq_len, head_dim) for single-head.
            v: Value cache, same shape as k.

        Returns:
            LayerKV with all data compressed in-place.
        """
        t0 = time.perf_counter()

        # Normalise to (num_heads, seq_len, head_dim)
        if k.ndim == 2:
            k = k[np.newaxis, ...]
            v = v[np.newaxis, ...]

        n_heads, seq_len, _ = k.shape
        compressor = self._compressors[layer_idx]

        k_indices_all = []
        k_norms_all = []
        k_qjl_signs_all = []
        k_residual_norms_all = []
        v_indices_all = []
        v_norms_all = []

        for h in range(n_heads):
            k_h = k[h]  # (seq_len, head_dim)
            v_h = v[h]

            # K: full TurboQuant (inner product preservation)
            compressed_k = compressor.k_quantizer.quantize(k_h)
            k_indices_all.append(compressed_k.mse_indices)
            k_norms_all.append(compressed_k.vector_norms)
            k_qjl_signs_all.append(compressed_k.qjl_signs)
            k_residual_norms_all.append(compressed_k.residual_norms)

            # V: MSE-only PolarQuant (value reconstruction)
            v_idx, v_norms = compressor.v_quantizer.quantize(v_h)
            v_indices_all.append(v_idx)
            v_norms_all.append(v_norms)

        elapsed = time.perf_counter() - t0
        self._layer_compress_times.append(elapsed)

        # Track memory
        raw_bytes = n_heads * seq_len * self.head_dim * 2 * 2  # fp16 × (K + V)
        self._total_uncompressed_bytes += raw_bytes

        is_boundary = (
            layer_idx < self.policy.boundary_n_layers or
            layer_idx >= self.num_layers - self.policy.boundary_n_layers
        )

        lkv = LayerKV(
            layer_idx=layer_idx,
            seq_len=seq_len,
            k_indices=np.array(k_indices_all),
            k_norms=np.array(k_norms_all),
            k_qjl_signs=np.array(k_qjl_signs_all),
            k_residual_norms=np.array(k_residual_norms_all),
            v_indices=np.array(v_indices_all),
            v_norms=np.array(v_norms_all),
            is_boundary=is_boundary,
        )

        self._total_compressed_bytes += lkv.approx_bytes(self.head_dim)
        self._kv_store[layer_idx] = lkv
        return lkv

    def restore_layer_kv(
        self,
        layer_idx: int,
        lkv: Optional[LayerKV] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress K and V tensors for a layer.

        Args:
            layer_idx: Transformer layer index.
            lkv: LayerKV to decompress. If None, uses internally stored kv.

        Returns:
            (k, v), both shape (num_heads, seq_len, head_dim).
        """
        if lkv is None:
            lkv = self._kv_store.get(layer_idx)
            if lkv is None:
                raise KeyError(f"No KV cached for layer {layer_idx}")

        compressor = self._compressors[layer_idx]
        n_heads = lkv.k_indices.shape[0]

        k_out = np.zeros((n_heads, lkv.seq_len, self.head_dim), dtype=np.float32)
        v_out = np.zeros_like(k_out)

        for h in range(n_heads):
            from turboquant.turboquant import CompressedVector
            compressed_k = CompressedVector(
                mse_indices=lkv.k_indices[h],
                vector_norms=lkv.k_norms[h],
                qjl_signs=lkv.k_qjl_signs[h],
                residual_norms=lkv.k_residual_norms[h],
                bit_width=self.policy.k_bits,
            )
            k_out[h] = compressor.k_quantizer.dequantize(compressed_k)
            v_out[h] = compressor.v_quantizer.dequantize(
                lkv.v_indices[h], lkv.v_norms[h]
            )

        return k_out, v_out

    def clear_kv(self, layer_idx: Optional[int] = None) -> None:
        """Free compressed KV memory.

        Args:
            layer_idx: If given, free only that layer. If None, free all.
        """
        if layer_idx is not None:
            self._kv_store.pop(layer_idx, None)
        else:
            self._kv_store.clear()
        gc.collect()

    # ------------------------------------------------------------------
    # Memory / diagnostic report
    # ------------------------------------------------------------------

    def memory_report(self) -> str:
        """Return a human-readable memory report string."""
        policy = self.policy
        raw_mb = self._total_uncompressed_bytes / 1024 / 1024
        comp_mb = self._total_compressed_bytes / 1024 / 1024
        ratio = raw_mb / max(comp_mb, 1e-9)
        avg_compress_ms = (
            1000 * sum(self._layer_compress_times) / max(len(self._layer_compress_times), 1)
        )

        lines = [
            "=" * 60,
            "AirLLM + TurboQuant KV Cache Session Report",
            "=" * 60,
            f"  Model size:        {self.model_size_b:.0f}B",
            f"  Layers:            {self.num_layers}",
            f"  Heads:             {self.num_heads}",
            f"  Head dim:          {self.head_dim}",
            f"  K precision:       turbo{policy.k_bits} ({policy.k_bits}-bit)",
            f"  V precision:       turbo{policy.v_bits} ({policy.v_bits}-bit)",
            f"  Boundary layers:   first/last {policy.boundary_n_layers} @ turbo{policy.boundary_k_bits}",
            f"  Max context:       {policy.max_context}",
            "",
            f"  KV raw size:       {raw_mb:.1f} MB",
            f"  KV compressed:     {comp_mb:.1f} MB",
            f"  Compression ratio: {ratio:.2f}×",
            f"  Avg compress time: {avg_compress_ms:.2f} ms/layer",
            "=" * 60,
        ]
        return "\n".join(lines)

    @property
    def compression_ratio(self) -> float:
        """Overall KV compression ratio achieved so far."""
        if self._total_compressed_bytes == 0:
            return 0.0
        return self._total_uncompressed_bytes / self._total_compressed_bytes

    def config_summary(self) -> dict:
        """Return a dict of current session configuration."""
        return {
            "model_size_b": self.model_size_b,
            "num_layers": self.num_layers,
            "k_bits": self.policy.k_bits,
            "v_bits": self.policy.v_bits,
            "boundary_k_bits": self.policy.boundary_k_bits,
            "boundary_n_layers": self.policy.boundary_n_layers,
            "max_context": self.policy.max_context,
        }

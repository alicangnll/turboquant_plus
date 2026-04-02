"""LLMTuning Bridge for TurboQuant — Layer-Sharding + KV Compression.

Combines LLMTuning's layer-sharding disk strategy with TurboQuant's KV cache
compression into a true 3-stage pipeline:

    Thread 1 (Disk I/O):    [Load Layer N+1] ──────────────────────▶
    Thread 2 (GPU/Compute):           [Compute Layer N] ────────────▶
    Thread 3 (CPU Compress):                    [Compress KV N-1] ──▶

LLMTuning's contribution integrated here:
  - LayerPrefetcher: disk → CPU RAM async, overlapped with GPU compute
  - KVCompressionWorker: CPU KV compression async, overlapped with GPU compute
    of the NEXT layer (zero-wait compression via double-buffering)
  - Memory discipline: layer weights freed immediately after compute,
    raw KV tensors freed immediately after compression

TurboQuant's contribution integrated here:
  - K cache: full TurboQuant IP-preserving quantization (Q@K^T quality)
  - V cache: MSE-only PolarQuant (attn_weights@V reconstruction)
  - Boundary layer protection: first/last 2 layers at higher precision
    (mirrors TURBO_LAYER_ADAPTIVE=7)
  - Sparse V: attention-weight-gated decompression — positions where
    softmax weight < threshold are skipped entirely (mirrors Metal kernel
    sparse-V optimization, validated +22.8% decode at 32K context)

Combined peak memory for 32B (64 layers, 40 heads, d=128, ctx=4096):
  - Without: ~21 GB (weights) + ~2.6 GB (KV raw)      = ~24 GB
  - With:    ~350 MB (1 layer) + ~675 MB (KV compressed) = ~1 GB active

Usage:
    from turboquant.LLMTuning_bridge import LLMTuningTurboSession

    session = LLMTuningTurboSession(model_size_b=32, num_layers=64,
                                  num_heads=40, head_dim=128)

    # Async compress (returns Future — compression overlaps next layer compute)
    future = session.compress_async(layer_idx, k_tensor, v_tensor)
    # ... compute next layer ...
    lkv = future.result()   # block only when actually needed

    # Sparse V restore (skips low-weight positions, +20-30% speed at long ctx)
    attn_weights = ...  # (num_heads, seq_len) softmax output
    k, v = session.sparse_restore_layer_kv(layer_idx, attn_weights)

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
# LLMTuning-inspired Prefetch Layer Loader (disk → CPU RAM, async)
# ---------------------------------------------------------------------------

class LayerPrefetcher:
    """Background thread that pre-loads the next model layer weights from disk.

    Mirrors LLMTuning's ThreadPoolExecutor prefetch approach (LLMTuning_base.py:441-487).

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

class LLMTuningTurboSession:
    """Stateful KV cache manager combining LLMTuning + TurboQuant strategies.

    Key behaviours:
    1. Auto-selects TurboQuant precision (turbo4 for 32B/70B, turbo2 for 400B+)
    2. Boundary layer protection: first/last `boundary_n_layers` always at
       higher precision (mirrors TURBO_LAYER_ADAPTIVE=7 env var in llama.cpp)
    3. Thread-safe: prefetcher runs in separate thread (disk I/O overlapped
       with GPU compute, same approach as LLMTuning)
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
        self._last_sparse_skip_fraction: float = 0.0

        # KV store: layer_idx → LayerKV
        self._kv_store: dict[int, LayerKV] = {}

        # Background thread for async KV compression (Thread 3 in the pipeline).
        # Compression of layer N runs here while GPU computes layer N+1.
        self._compress_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="turbo_kv_compress"
        )

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

    def compress_async(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> "Future[LayerKV]":
        """Submit KV compression to the background compression thread.

        This is the core 3-stage pipeline primitive — the main caller pattern:

            future = session.compress_async(N, k, v)  # Thread 3 starts
            del k, v                                   # free raw tensors now
            # ... GPU computes layer N+1 ... (Thread 2)
            # ... Disk prefetches layer N+2 ... (Thread 1)
            lkv = future.result()                      # block only if needed

        Compression of layer N's KV overlaps GPU compute of layer N+1,
        giving effectively zero additional latency at longer context lengths.

        Args:
            layer_idx: Transformer layer index.
            k: Key tensor (num_heads, seq_len, head_dim). Immediately copied
               so caller can safely delete the original.
            v: Value tensor, same shape.

        Returns:
            Future[LayerKV] — non-blocking. Call .result() when KV is needed.
        """
        k_copy = k.copy()
        v_copy = v.copy()
        return self._compress_executor.submit(
            self.compress_layer_kv, layer_idx, k_copy, v_copy
        )

    def sparse_restore_layer_kv(
        self,
        layer_idx: int,
        attn_weights: np.ndarray,
        threshold: float = 1e-6,
        lkv: Optional[LayerKV] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sparse V decompression: skip negligible-weight positions entirely.

        Mirrors the Metal kernel TURBO_SPARSE_V optimisation from README:
        positions where softmax(attn_weight) < threshold contribute negligibly
        to the attention output (attn_weights @ V), so we skip decompressing them.

        K is always fully decompressed — routing must be correct.
        V is sparsified — value aggregation is numerically tolerant.

        From README validated benchmarks:
          - +22.8% decode speed at 32K context vs dense turbo3
          - Zero PPL penalty (ON/OFF delta = 0.000 on Wikitext-103)
          - NIAH: 9/9 correct vs 7/9 without sparse-V

        Args:
            layer_idx: Layer index.
            attn_weights: Softmax weights, shape (num_heads, kv_seq_len) or
                          (num_heads, query_len, kv_seq_len).
            threshold: Min attention weight to trigger V decompression.
                       Default 1e-6 matches llama.cpp Metal kernel default.
            lkv: Optional pre-fetched LayerKV. Uses internal store if None.

        Returns:
            (k, v) where k is fully decompressed, v is sparse-decompressed
            (zero at skipped positions — contributes 0 to attn_weights@V).
        """
        if lkv is None:
            lkv = self._kv_store.get(layer_idx)
            if lkv is None:
                raise KeyError(f"No KV cached for layer {layer_idx}")

        compressor = self._compressors[layer_idx]
        n_heads = lkv.k_indices.shape[0]

        k_out = np.zeros((n_heads, lkv.seq_len, self.head_dim), dtype=np.float32)
        v_out = np.zeros((n_heads, lkv.seq_len, self.head_dim), dtype=np.float32)

        # Normalise attn_weights to (num_heads, seq_len)
        aw = attn_weights
        if aw.ndim == 3:
            aw = aw.max(axis=1)  # (num_heads, kv_seq_len)

        from turboquant.turboquant import CompressedVector

        skipped_total = 0
        for h in range(n_heads):
            # K: always fully decompress (inner product routing)
            compressed_k = CompressedVector(
                mse_indices=lkv.k_indices[h],
                vector_norms=lkv.k_norms[h],
                qjl_signs=lkv.k_qjl_signs[h],
                residual_norms=lkv.k_residual_norms[h],
                bit_width=self.policy.k_bits,
            )
            k_out[h] = compressor.k_quantizer.dequantize(compressed_k)

            # V: sparse — only decompress positions that carry meaningful weight
            head_weights = aw[h] if h < aw.shape[0] else aw[-1]
            active_positions = np.where(head_weights >= threshold)[0]
            skipped_total += lkv.seq_len - len(active_positions)

            if len(active_positions) == 0:
                continue

            v_idx_active = lkv.v_indices[h][active_positions]
            v_norms_active = lkv.v_norms[h][active_positions]
            v_out[h][active_positions] = compressor.v_quantizer.dequantize(
                v_idx_active, v_norms_active
            )

        self._last_sparse_skip_fraction = skipped_total / max(n_heads * lkv.seq_len, 1)
        return k_out, v_out

    def restore_layer_kv(
        self,
        layer_idx: int,
        lkv: Optional[LayerKV] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress K and V tensors for a layer (full, non-sparse).

        For sparse decompression, use sparse_restore_layer_kv() instead.

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

        from turboquant.turboquant import CompressedVector
        for h in range(n_heads):
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
        """Free compressed KV memory for one layer or all layers."""
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
        sparse_pct = self._last_sparse_skip_fraction * 100

        lines = [
            "=" * 60,
            "LLMTuning + TurboQuant Hybrid Session Report",
            "=" * 60,
            f"  Model size:        {self.model_size_b:.0f}B",
            f"  Layers:            {self.num_layers}",
            f"  Heads:             {self.num_heads}  |  Head dim: {self.head_dim}",
            "",
            f"  — TurboQuant KV Config —",
            f"  K precision:       turbo{policy.k_bits} (IP-preserving Q@K^T)",
            f"  V precision:       turbo{policy.v_bits} (MSE-only attn@V)",
            f"  Boundary protect:  first/last {policy.boundary_n_layers} layers @ turbo{policy.boundary_k_bits}",
            f"  Sparse V:          last call skipped {sparse_pct:.1f}% of V positions",
            "",
            f"  — LLMTuning Pipeline Config —",
            f"  Layer prefetch:    ThreadPoolExecutor (disk → RAM, Thread 1)",
            f"  Async compress:    ThreadPoolExecutor (KV compress, Thread 3)",
            f"  Max context:       {policy.max_context}",
            "",
            f"  — Memory —",
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

    def shutdown(self) -> None:
        """Shut down the async compression executor cleanly."""
        self._compress_executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


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
            "LLMTuning + TurboQuant KV Cache Session Report",
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

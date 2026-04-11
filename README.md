# TurboTuning
### Extreme-Efficiency Inference Engine for Large Language Models

> **TurboTuning** is a system-level inference framework that enables 70B+ parameter language models to run on commodity hardware through two orthogonal techniques: **TurboQuant+** [[1]](https://github.com/TheTom/turboquant_plus) (KV cache compression via Walsh-Hadamard rotation and scalar quantization) and **LLMTuning** (model-weight memory virtualization via OS-level page management). The two modules are independent and composable.

[Architecture & Design](MAP.md) | [Memory Analysis](docs/memory-rss-targets.md) | [Roadmap](PLAN.md)

---

## Abstract

Running large language models (LLMs) locally is constrained by two independent bottlenecks: (1) the KV cache, which grows quadratically with context length and consumes substantial VRAM, and (2) model weights, which at 70B parameters exceed the physical RAM of most workstations. TurboTuning addresses each bottleneck with a dedicated module. TurboQuant+ applies a Walsh-Hadamard Transform (WHT) to attention keys and values before quantizing them to 2–4 bits, reducing KV memory by 2.5–6.4× with negligible perplexity degradation. LLMTuning virtualizes model weights by treating physical RAM as a sliding window over SSD-resident data, keeping only 1–2 transformer layers resident at any moment through predictive `madvise` paging and a double-buffered async prefetch pipeline. Together, they enable sustained inference of 70B-class models on systems with 16–24 GB of unified memory.

---

## The Two Modules

| | TurboQuant+ | LLMTuning |
|---|---|---|
| **Target** | KV cache (attention keys & values) | Model weights (transformer layers) |
| **Mechanism** | WHT rotation + 2/3/4-bit scalar quantization | OS page eviction (`madvise`) + async prefetch |
| **Primary savings** | VRAM / KV memory | Physical RAM (peak RSS) |
| **Activation** | `--cache-type-k turbo* --cache-type-v turbo*` | Auto-triggered by any turbo cache type |
| **Composable?** | Yes — runs without LLMTuning | Yes — runs with any KV type |
| **Runtime overhead** | WHT O(d log d) per K/V write | Background I/O thread; ~0 GPU stall |

---

## Module 1 — TurboQuant+ (KV Cache Compression)

TurboQuant+ implements the **PolarQuant** algorithm [[2]](https://arxiv.org/abs/2504.19874). It operates on the attention key and value tensors written to the KV cache during each forward pass.

### Step 1: Walsh-Hadamard Transform

Raw transformer activations are heavy-tailed and spiky — difficult to quantize accurately. Multiplying each K or V vector by a random Hadamard matrix H ∈ ℝ^(d×d) (applied via the Fast Walsh-Hadamard Transform, O(d log d)) rotates the distribution into an approximately Gaussian shape. This is the **incoherence** property used in compressed sensing:

```
K_rotated = H · K_natural
```

After rotation, scalar quantization at 2–4 bits produces substantially lower reconstruction error than quantizing raw activations.

### Step 2: Scalar Quantization (turbo2 / turbo3 / turbo4)

| Type | Bits/element | Memory vs f16 | Storage domain | Best for |
|---|---|---|---|---|
| `turbo2` | 2 | ~15.6% | WHT-rotated | Maximum compression |
| `turbo3` | 3 | ~20.4% | WHT-rotated | Balanced quality/memory |
| `turbo4` | 4 | ~26.6% | **Natural** | Default; best quality |

**Domain asymmetry (turbo4):** turbo2 and turbo3 store activations in WHT-rotated space. turbo4 stores them in natural space (WHT is skipped during quantization). `llama-graph.cpp` detects `k->type` at runtime and conditionally applies WHT to the query Q before computing Q·K^T, ensuring the inner product is always taken in a consistent domain.

### Step 3: GPU Dequantization (Metal / CUDA)

Dequantization occurs inside Flash Attention kernels. On Apple Silicon, 537 Metal kernel specializations cover all K/V type combinations (`kernel_flash_attn_ext_kturbo4_vturbo4`, `_kturbo4_vturbo2`, etc.). The non-vectorized FA path is always selected when any turbo type is active. CPU fallback is provided by `ggml-turbo-quant.c`.

### Standalone usage

```bash
./build/bin/llama-cli \
    -m model.gguf -ngl 99 -c 4096 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3
```

Asymmetric K/V types are supported. K is less precision-sensitive (used only in the dot product); V is more sensitive (linearly combined into the output). Recommended: `K=turbo4, V=turbo3` for quality; `K=turbo4, V=turbo2` for maximum savings.

---

## Module 2 — LLMTuning (Weight Memory Virtualization)

LLMTuning treats physical RAM as a fixed-capacity cache in front of the SSD, with transformer layers as the cache lines. It is implemented in `llama-llmtuning.cpp` as a three-stage concurrent pipeline.

### Stage 1: Cold Boot Evacuation

After the model file is `mmap`'d into virtual address space, LLMTuning immediately issues `madvise(MADV_DONTNEED)` across all transformer layer weight pages. The kernel releases the corresponding physical frames. Virtual addresses remain valid — the OS will fault pages back in from the mmap'd file on demand. Initial physical RAM footprint for a 70B model: ~1.1 GB (embedding + output layers only).

### Stage 2: Predictive Paging — Double-Buffered Prefetch

A pool of **2 worker threads** services a ring-buffer job queue of depth `PREFETCH_LOOKAHEAD = 2`. For each layer N being computed, the main thread enqueues prefetch requests for layers N+1 and N+2 concurrently:

```
GPU: [Compute N  ]  [Compute N+1]  [Compute N+2]
I/O: [Fetch N+1,N+2]    [Fetch N+2,N+3]    ...
```

Each worker issues `madvise(MADV_WILLNEED)` on the target layer's tensor pages, causing the kernel to schedule async disk reads into the page cache. On **Linux**, `MADV_WILLNEED` is supplemented with `posix_fadvise(POSIX_FADV_WILLNEED)` at the VFS layer.

A **memory pressure guard** checks `llama_get_free_ram_mb()` before each enqueue. If free RAM drops below `pressure_threshold_mb` (adaptive: 1 GB on ≤4 GB systems, 2 GB otherwise), the farther lookahead slot (N+2) is silently dropped, preventing thrashing.

### Stage 3: Active Eviction

Immediately after the GPU finishes layer N, `madvise` is called with the eviction flag on that layer's pages:

- **macOS**: `MADV_DONTNEED` — immediate physical page release (safe with Metal's unified memory buffer model, which holds its own reference)
- **Linux**: `MADV_FREE` — lazy release; kernel reclaims pages only under pressure, avoiding a disk re-read if the same layer is accessed again soon

Steady-state physical RAM occupied by weights ≈ 2–3 layers × layer_size. For a 70B Q4_K_M model: ~2–3 × 300 MB ≈ 600–900 MB.

### Stage 4: Async KV Compression (Background)

A third worker thread (`llama_kv_compress_worker`) processes KV tensors after each layer's GPU compute completes. When the KV cache is allocated as f16 (e.g., for first-token prefill), the compressor performs in-place quantization:

1. Convert f16 → f32 staging buffer
2. Quantize: K → `TURBO4_0`, V → `TURBO3_0` (asymmetric by default)
3. Write compressed bytes back into the original buffer (always fits; turbo types are smaller than f16)
4. Update `t->type` and stride metadata

This converts f16 KV entries to compressed format asynchronously, overlapping with subsequent layer computation. Metal/CUDA buffers are detected via `ggml_backend_buffer_is_host()` and skipped.

### TQR — Page-Aligned Weight Cache

On first run, LLMTuning saves a `.tqr` (TurboQuant Repack) file alongside the model. TQR stores weight tensors at page-aligned offsets, enabling `mmap` with zero-copy and eliminating the repack buffer that standard GGUF loading allocates at boot. Subsequent runs hot-swap weights from TQR, reducing cold-start time from ~5–8 s to ~1.1 s for a 70B model. Stale `.tqr` files from other models are auto-cleaned at session init.

### Memory timeline (steady state)

```
T=0  Boot:    mmap model → MADV_DONTNEED all layers   → RSS ≈ 1.1 GB
T=1  Layer 0: Prefetch 1, 2 (async)                   → RSS ≈ 1.4 GB
              GPU computes 0
              Evict 0                                  → RSS ≈ 1.1 GB
T=2  Layer 1: Already prefetched                       → RSS ≈ 1.4 GB
              GPU computes 1
              Evict 1, prefetch 3                      → RSS ≈ 1.1 GB
...
```

### Auto-activation

LLMTuning activates automatically when any `--cache-type-k turbo*` or `--cache-type-v turbo*` flag is present. No separate flag required. The argument parser sets `turbo_async = true`, which causes `llama_context` to instantiate a `llama_tuning_session` at context creation.

---

## Combined: TurboTuning Pipeline

When both modules are active, memory savings are multiplicative:

| Scenario | Weight RSS | KV Memory | Total (8B, 4K ctx) |
|---|---|---|---|
| Baseline llama.cpp | Full (~4.7 GB) | ~2.1 GB f16 | ~6.8 GB |
| TurboQuant+ only | Full (~4.7 GB) | ~0.56 GB turbo4 | ~5.3 GB |
| LLMTuning only | ~0.7 GB active | ~2.1 GB f16 | ~2.8 GB |
| **TurboTuning (both)** | **~0.7 GB active** | **~0.56 GB turbo4** | **~1.3 GB** |

For a 70B model (Q4_K_M, ~40 GB on disk):

| Config | Required RAM | Feasible on |
|---|---|---|
| Standard llama.cpp | ~42 GB | Mac Studio M2 Ultra only |
| TurboQuant+ only | ~42 GB weights + compressed KV | Mac Studio M2 Ultra |
| **TurboTuning** | **~2–3 GB active + compressed KV** | **MacBook Pro M3 (16 GB)** |

---

## Benchmark
![Benchmark Results](benchmark/benchmark_results_8b.png)

## Quick Start

### Build

```bash
cd llama-cpp-turboquant

# macOS (Metal + OpenMP)
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli

# Linux (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli
```

### Run — TurboQuant+ only (model fits in RAM)

```bash
./build/bin/llama-cli \
    -m model.gguf -ngl 99 -c 4096 \
    --cache-type-k turbo4 --cache-type-v turbo3 \
    -cnv -sys "You are a helpful assistant."
```

### Run — Full TurboTuning (model larger than RAM)

```bash
# macOS
./run_turboquant_demo_macos.sh

# Linux
./run_turboquant_demo_linux.sh
```

The demo scripts compile the engine, present a model menu, download if needed, run a validation pass (50-token generation), and launch the interactive session.

---

## Supported Models

| Model | Disk size | Notes |
|---|---|---|
| Llama 3.1 8B Instruct Q4_K_M | ~5 GB | Default; fits 16 GB RAM |
| Qwen 2.5 32B Instruct Q4_K_M | ~20 GB | Needs 24 GB+ with TurboTuning |
| Command R+ 104B Q2_K | ~43 GB | Needs 16 GB+ with TurboTuning |
| GPT 20B Q4_K_M | ~12 GB | No chat template |
| Gemma 4 31B Q4_K_M | ~18 GB | macOS only |
| Qwen 2.5 Coder 7B Q4_K_M | ~5 GB | Code tasks |

---

## References

1. Turney, T. (2026). *TurboQuant+: Extreme-Efficiency Inference Engine for Large Language Models*. GitHub repository. [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
2. *PolarQuant Algorithm*. ICLR 2026. (arXiv:2504.19874)

---

## License

Apache 2.0. Based on [llama.cpp](https://github.com/ggml-org/llama.cpp).

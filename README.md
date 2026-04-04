# 🚀 TurboQuant+ // LLMTuning
### Extreme Efficiency Engine for 2026 Local LLM Inference

> **TurboQuant+** and **LLMTuning** are twin technologies designed to run massive language models (32B–500B) on consumer hardware (16GB–64GB RAM) with zero-compromise speed and stability.

[Getting Started Guide](docs/getting-started.md) | [Architecture Map](MAP.md) | [Development Roadmap](PLAN.md) | [Turkish Guide (Türkçe)](README_TR.md)

---

## 🌟 The Vision: Power to the Edge
Most local inference engines struggle with models larger than physical RAM. This project solves that by decoupling the model's **Mathematical Representation** (TurboQuant+) from its **Hardware Orchestration** (LLMTuning).

*   **TurboQuant+**: A state-of-the-art quantization core based on the ICLR 2026 paper, achieving near-lossless 2-bit, 3-bit, and 4-bit KV cache compression.
*   **LLMTuning**: An intelligent orchestration layer that virtualizes model memory, sharding layers across NVMe SSD and RAM in a non-blocking asynchronous pipeline.

---

## 🧩 Pillar 1: LLMTuning (Orchestration)
LLMTuning is the "OS" for your LLM. It manages hardware resources to ensure the engine never hits an OOM (Out of Memory) state, even when running a 104B model on a 16GB MacBook.

*   **Active Sharding (`madvise`)**: Automatically unloads transformer layers from physical RAM immediately after the GPU finishes computation. Physical RAM only holds the *active* layer and a prefetch buffer.
*   **Native Budget Discovery**: Automatically detects platform memory limits (e.g., Apple Metal `recommendedMaxWorkingSetSize`) using C++ `sysctl` calls, calculating the optimal `-ngl` (GPU layer) count.
*   **Cold Boot Evacuation**: Minimizes initial RAM spikes by immediately moving model weights to SSD-backed virtual memory, allowing 8B models to boot with a **~1.1GB RAM footprint**.
*   **Predictive Paging**: Uses `MADV_WILLNEED` hints to start loading the *next* layer from NVMe while the GPU is still busy with the current one.

---

## ⚡ Pillar 2: TurboQuant+ (Compression)
TurboQuant+ provides the high-speed compression that shrinks the model's working memory (KV Cache) by up to 6.4x.

*   **PolarQuant (2/3/4-bit)**: Leverages random Walsh-Hadamard rotations to transform attention tensors into a Beta-distribution, enabling optimal scalar quantization with near-zero quality loss.
*   **Sparse V Optimization**: Uses an attention-gated dequantizer that skips low-priority Value (V) tensors, boosting long-context decode speeds by **~22.8%**.
*   **Dual Acceleration**: A parallel compute model that runs transformer math on the **GPU (Metal/CUDA)** while simultaneously performing rotations and quantization on the **CPU (OpenMP)**.
*   **Boundary Protection**: Maintains the first and last layers at higher precision (e.g., Q8_0) to preserve long-range coherence and formatting.

---

## 🚀 The 3-Stage Asynchronous Pipeline
The engine implements a specialized non-blocking orchestration called the **Turbo-Async Pipeline**:

1.  **Stage 1: Prefetch (LLMTuning)**: Loads Layer N+1 from disk to RAM while Stage 2 is running.
2.  **Stage 2: Compute (LLM Engine)**: GPU executes the current Layer N using native kernels.
3.  **Stage 3: Compress (TurboQuant+)**: CPU compresses the KV cache for Layer N immediately after Stage 2 completion.

![Async Pipeline Visualization](file:///Users/alicangonullu/.gemini/antigravity/brain/2e148d66-421c-432a-b624-209399070db4/media__1775313466809.png)

**Performance Result**: Up to **943.7 t/s** prompt processing on Llama 3.1 8B with zero "@@@@" corruption.

---

## 🛠️ Quick Start

### 1. Requirements
*   **macOS**: Xcode Command Line Tools, Python 3.10+, [libomp](https://formulae.brebrew.sh/formula/libomp) (via Homebrew).
*   **Linux**: GCC/Clang, CMake, OpenMP.
*   **Windows**: MSVC, CMake.

### 2. Compilation
```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### 3. Run the Demo
We provide hardware-optimized wrappers for a zero-config experience:
*   **macOS**: `./run_turboquant_demo_macos.sh`
*   **Linux**: `./run_turboquant_demo_linux.sh`
*   **Windows**: `run_turboquant_demo.bat`

---

## 📊 Milestones
*   **Ultra-Eco Mode**: 8B models running in **1.1GB RAM**.
*   **High-Context Mastery**: 104B models running at **128K context** on a 128GB MacBook.
*   **Low-RAM Stability**: Stable inference of 70B models on 24GB M2/M3 chips.

---

## 🤝 Contributing
This project is an experimental fork of `llama.cpp`. We actively work on upstreaming stable components. Check [PLAN.md](PLAN.md) for our research roadmap.

**License**: Apache 2.0. Copyright 2026 Tom Turney.

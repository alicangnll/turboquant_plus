# Getting Started with TurboQuant+

Get up and running with TurboQuant+ KV cache compression in under 5 minutes.

## Prerequisites

- A GGUF model (download from [HuggingFace](https://huggingface.co))
- CMake 3.14+
- One of: Apple Silicon Mac, NVIDIA GPU (CUDA), AMD GPU (ROCm/HIP)

## Build

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# AMD (ROCm/HIP)
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Windows (CUDA, use Developer Command Prompt or WSL2)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

## Quick Start

Run a model with TurboQuant+ KV cache compression:

```bash
# Interactive chat
./build/bin/llama-cli -m model.gguf -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192

# Server mode
./build/bin/llama-server -m model.gguf -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192
```

## Choosing Your Config

### Safe default (works on any model)

```bash
-ctk q8_0 -ctv turbo4 -fa on
```

Asymmetric: full precision K, compressed V. Safe on all tested models including sensitive ones like Qwen2.5.

### More compression (works on most models)

```bash
-ctk q8_0 -ctv turbo3 -fa on
```

5.12x V compression with minimal quality loss (+1-2% PPL).

### Maximum compression (validated large models only)

```bash
-ctk turbo3 -ctv turbo3 -fa on
```

Symmetric. Works great on Llama 70B, Command-R+ 104B, Mistral 24B, Qwen3.5 MoE. **Do NOT use on Qwen2.5 with Q4_K_M weights** (catastrophic PPL).

### When to use what

| Your model | Recommended config |
|---|---|
| Q8_0 weights, any size | `-ctk turbo4 -ctv turbo4` or `-ctk turbo3 -ctv turbo3` |
| Q4_K_M, unknown model | `-ctk q8_0 -ctv turbo4` (safe default) |
| Q4_K_M, Qwen2.5 | `-ctk q8_0 -ctv turbo3` (must be asymmetric) |
| Q4_K_M, Llama/Mistral/Cohere 24B+ | `-ctk turbo3 -ctv turbo3` (symmetric works) |

For full details see [Configuration Recommendations](turboquant-recommendations.md).

## Built-in Optimizations

These features are enabled by default on recent builds. No extra flags needed.

### Sparse V Dequant (Metal only)

Skips V dequantization for positions where the softmax attention weight is negligible (< 1e-6). At long context, 90%+ of positions have negligible weight, so this eliminates roughly half the total dequant cost.

- +22.8% decode throughput at 32K context on MoE models
- Zero perplexity impact (validated at 32K, 50 chunks, wikitext-103)
- Works with any KV cache type (turbo3, q8_0, q4_0)
- Opt-out: `TURBO_SPARSE_V=0`

See [Sparse V paper](papers/sparse-v-dequant.md).

### Boundary V (auto-enabled with turbo2-V)

Protects the first 2 + last 2 transformer layers with q8_0-V while compressing all remaining layers with turbo2-V. Recovers 37-91% of the quality gap between turbo2 and turbo3, with no speed penalty.

- Auto-enabled when you use `-ctv turbo2`
- Opt-out: `TURBO_LAYER_ADAPTIVE=0`
- Manually enable on older builds: `TURBO_LAYER_ADAPTIVE=7`

See [Boundary V paper](papers/layer-aware-v-compression.md).

### Maximum V Compression

```bash
# turbo2-V with Boundary V (auto-enabled)
# Best for extreme memory pressure. +5-9.5% PPL.
./build/bin/llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa on -ngl 99
```

## Benchmarking

Help the project by sharing your numbers. Here's how to generate comparable results.

### Step 1: Download test data

```bash
# Wikitext-2 for perplexity testing
wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet
# Or use the raw text version from the dataset page
```

### Step 2: Run baseline (always do this first)

```bash
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk q8_0 -ctv q8_0 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20
```

### Step 3: Run turbo configs

```bash
# Asymmetric (safe for any model)
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk q8_0 -ctv turbo3 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20

# Symmetric (if your model supports it)
./build/bin/llama-perplexity \
  -m model.gguf \
  -ctk turbo3 -ctv turbo3 \
  -fa on -ngl 99 \
  -f wiki.test.raw \
  -c 512 --chunks 20
```

### Step 4: Speed benchmarks

```bash
# Short context
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1

# Long context (compare turbo3 vs q8_0 at each length)
./build/bin/llama-bench -m model.gguf -ctk q8_0 -ctv q8_0 -fa 1 -p 8192 -r 1
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1 -p 8192 -r 1
./build/bin/llama-bench -m model.gguf -ctk q8_0 -ctv q8_0 -fa 1 -p 32768 -r 1
./build/bin/llama-bench -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1 -p 32768 -r 1
```

### Step 5: Share results

Post your numbers in the [GitHub discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969) or open an issue on the [repo](https://github.com/TheTom/llama-cpp-turboquant).

Include: model name, weight quantization, GPU, VRAM, turbo config, PPL, and speed numbers.

## Apple Silicon: Large Models at Long Context

If you're running 70B+ models at long context on Apple Silicon, macOS caps GPU memory at ~75% of RAM by default. This causes Metal to hang at ~49K context. Fix:

```bash
# Recommended: 90% of physical RAM (safe for sustained inference)
# 128GB Mac
sudo sysctl iogpu.wired_limit_mb=117964

# 96GB Mac
sudo sysctl iogpu.wired_limit_mb=88474

# 64GB Mac
sudo sysctl iogpu.wired_limit_mb=58982
```

No reboot required. Resets on reboot.

## Resources

- [Configuration Recommendations](turboquant-recommendations.md) (full config guide with all tested models)
- [M5 Max Stress Test](papers/m5-max-stress-test.md) (70B and 104B results)
- [Sparse V Dequantization](papers/sparse-v-dequant.md)
- [Boundary V](papers/layer-aware-v-compression.md)
- [Block Size Optimization](papers/block-size-experiment.md)
- [TurboQuant paper (Google Research)](https://arxiv.org/abs/2504.19874)

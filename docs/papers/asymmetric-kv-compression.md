# Asymmetric K/V Cache Compression: Why V is Free and K is Everything

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

The TurboQuant paper (ICLR 2026) tests KV cache compression with symmetric configurations (same quantization for both K and V). In practice, symmetric turbo quantization is catastrophic on certain model families with low-bit weight quantization (Q4_K_M). We show that this failure is entirely K-side: V compression has zero measurable effect on attention quality when K precision is maintained.

This asymmetry arises from the attention mechanism itself. K determines attention routing via softmax, which amplifies small errors exponentially. V errors scale linearly through the weighted sum. On Qwen2.5-7B Q4_K_M, symmetric turbo3 produces PPL 3,556 (catastrophic), while asymmetric q8_0-K + turbo3-V produces PPL 6.71 (+2.0% vs baseline). Same total compression budget, opposite quality outcomes.

We validate this finding across 7 models (1.5B to 104B), 3 weight quantizations (Q4_K_M, Q6_K, Q8_0), 3 GPU backends (Metal, CUDA, HIP), and 2 Apple Silicon generations (M2 Pro, M5 Max). The practical recommendation is simple: compress V maximally, spend bits on K.

---

## 1. Introduction

KV cache compression reduces memory consumption during LLM inference, enabling longer context windows and larger models on consumer hardware. Google's TurboQuant (ICLR 2026) achieves 4.6-5.12x compression via Walsh-Hadamard rotation and polar quantization. The paper validates quality using fp16 model weights.

In practice, most users run quantized model weights (Q4_K_M, Q6_K) to fit models in limited VRAM. When TurboQuant KV cache compression is applied on top of already-quantized weights, the errors compound. We refer to this as **quantization stacking**: weight quantization error + KV cache quantization error.

The question is: does stacking affect K and V equally?

---

## 2. The Asymmetry: Softmax Amplification

The attention mechanism computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

K determines **which tokens** receive attention weight via softmax. V determines **what information** flows from those tokens. These are fundamentally different roles:

- **K errors** shift the Q*K dot product scores. Softmax amplifies small score differences exponentially: a perturbation of $\epsilon$ in logit space produces an $O(e^\epsilon)$ change in the attention distribution. On sensitive models, this can flip which tokens dominate the output.

- **V errors** scale linearly through the weighted sum. If a V element is off by $\epsilon$, the output error is bounded by $w_i \cdot \epsilon$ where $w_i$ is the attention weight for that position. For non-dominant positions (most of them at long context), $w_i \approx 0$ and the error vanishes.

This predicts that K precision should matter far more than V precision for output quality. Our experiments confirm this.

---

## 3. Results

All PPL measured on wikitext-2-raw (512 context, 20 chunks) unless noted. All tests on Metal with flash attention, full GPU offload.

### 3.1 The Catastrophic Case: Qwen2.5-7B Q4_K_M

This model demonstrates the asymmetry most dramatically.

| K | V | M5 Max PPL | M2 Pro PPL | vs q8_0 | Status |
|---|---|-----------|-----------|---------|--------|
| q8_0 | q8_0 | 6.577 | 6.579 | baseline | healthy |
| q8_0 | turbo4 | 6.644 | 6.603 | +1.0% | healthy |
| q8_0 | turbo3 | 6.707 | 6.715 | +2.0% | healthy |
| q8_0 | turbo2 | 6.911 | — | +5.1% | healthy |
| turbo4 | turbo4 | 217.7 | 227.5 | catastrophic | avoid |
| turbo3 | turbo3 | 3,556 | 3,778 | catastrophic | avoid |

**V compression is essentially free**: q8_0-K + turbo2-V (2-bit V, 6.4x V compression) gives +5.1% PPL. The model is fully coherent.

**K compression is catastrophic**: turbo4-K + turbo4-V (same bits on both) gives PPL 218. The model is incoherent.

**Same total bits, opposite results.** The difference is 500x in PPL depending on which cache you compress.

Cross-hardware matched: M2 Pro and M5 Max produce equivalent quality patterns.

### 3.2 Qwen2.5-1.5B Q4_K_M

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| turbo3 | turbo3 | 8,641+ | catastrophic | avoid |

Symmetric turbo is catastrophic on Qwen2.5 at all sizes, not just 7B. The sensitivity is architecture-dependent.

### 3.3 phi-4-14B (Q8_0 Weights)

| K | V | M5 Max PPL | vs q8_0 | Status |
|---|---|-----------|---------|--------|
| q8_0 | q8_0 | 4.690 | baseline | healthy |
| q8_0 | turbo4 | 4.702 | +0.3% | healthy |
| q8_0 | turbo3 | 4.742 | +1.1% | healthy |
| q8_0 | turbo2 | 4.835 | +3.1% | healthy |
| turbo4 | turbo4 | 4.770 | +1.7% | healthy |
| turbo3 | turbo3 | 4.886 | +4.2% | healthy |

With Q8_0 weights, both symmetric and asymmetric work. The quantization stacking effect only appears when weight quantization is already aggressive (Q4_K_M). V compression gradient is clear: turbo4-V +0.3%, turbo3-V +1.1%, turbo2-V +3.1%. All healthy.

### 3.4 Mistral-Small-24B Q4_K_M

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| turbo3 | turbo3 | 4.987 | healthy | healthy |

Q4_K_M symmetric turbo is not universally broken. Mistral-24B handles it fine. The sensitivity is model-family-dependent, not purely a function of weight quantization.

### 3.5 Llama-3.1-70B Q4_K_M

| K | V | PPL | vs q8_0 (3.257) | Status |
|---|---|-----|---------|--------|
| q8_0 | turbo4 | 3.301 | +1.3% | healthy |
| q8_0 | turbo3 | 3.325 | +2.1% | healthy |
| q8_0 | turbo2 | 3.568 | +9.5% | healthy |
| turbo4 | turbo4 | 3.461 | +6.3% | healthy |
| turbo3 | turbo3 | 3.629 | +11.4% | usable |
| turbo2 | turbo2 | 5.161 | +58.5% | degraded |

**Asymmetric rescue**: q8_0/turbo2 = +9.5% vs symmetric turbo2/turbo2 = +58.5%. 6x improvement in quality degradation from the same V format, just by preserving K precision.

Llama-70B tolerates symmetric turbo3 (+11.4%), but asymmetric q8_0/turbo3 is better (+2.1%). Larger models absorb quantization stacking better than smaller ones.

### 3.6 Command-R+ 104B Q4_K_M

| K | V | PPL | vs q8_0 (6.192) | Status |
|---|---|-----|---------|--------|
| q8_0 | turbo4 | 6.211 | +0.3% | healthy |
| q8_0 | turbo3 | 6.296 | +1.7% | healthy |
| q8_0 | turbo2 | 6.678 | +7.9% | healthy |
| turbo4 | turbo4 | 6.312 | +1.9% | healthy |
| turbo3 | turbo3 | 6.415 | +3.6% | healthy |
| turbo2 | turbo2 | 7.049 | +13.8% | usable |

104B tolerates symmetric turbo even better than 70B. turbo4/turbo4 is +1.9%, nearly lossless. The trend is clear: bigger models have more capacity to absorb quantization stacking.

### 3.7 AMD HIP Validation (RX 9070 XT)

| K | V | PPL | vs q8_0 (7.794) | Status |
|---|---|-----|---------|--------|
| q8_0 | turbo4 | 7.876 | +1.0% | healthy |
| turbo4 | turbo4 | 401.4 | catastrophic | avoid |
| turbo3 | turbo3 | 81,277 | catastrophic | avoid |

Asymmetric q8_0/turbo4 confirmed on AMD. Symmetric Q4_K_M failure is consistent across all three GPU vendors (Metal, CUDA, HIP).

---

## 4. The Quantization Stacking Model

Why do some models survive symmetric turbo on Q4_K_M while others don't?

The hypothesis: K cache quantization error compounds multiplicatively with weight quantization error in the attention logit computation. The Q*K dot product involves both quantized-weight Q projections and quantized-cache K values. When both are lossy, the logit errors can exceed softmax's tolerance threshold.

The evidence supports this:

| Model | Weights | Symmetric turbo3 PPL | Status |
|-------|---------|:--------------------:|--------|
| Qwen2.5-1.5B | Q4_K_M | 8,641 | catastrophic |
| Qwen2.5-7B | Q4_K_M | 3,556 | catastrophic |
| Mistral-24B | Q4_K_M | 4.987 | healthy |
| Llama-70B | Q4_K_M | 3.629 | healthy |
| Command-R+ 104B | Q4_K_M | 6.415 | healthy |

The sensitivity is model-family-dependent. Qwen2.5 is sensitive at all sizes. Llama, Mistral, and Cohere are tolerant. Within tolerant families, larger models absorb stacking better (104B: +3.6% vs 70B: +11.4%).

V compression adds negligible error regardless of weight quantization. This is because V errors don't pass through softmax and don't interact multiplicatively with weight quantization in the same way.

---

## 5. Practical Recommendations

### For any model you haven't tested

```bash
# Safe default: asymmetric, full K precision, compressed V
llama-server -m model.gguf -ctk q8_0 -ctv turbo4 -fa on
```

### For models validated as symmetric-tolerant

```bash
# Symmetric: more compression, validated on Llama/Mistral/Cohere 24B+
llama-server -m model.gguf -ctk turbo3 -ctv turbo3 -fa on
```

### For maximum V compression

```bash
# Asymmetric with turbo2-V (Boundary V auto-enabled)
llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa on
```

### Decision tree

1. Q8_0+ weights? Symmetric turbo works. Use `-ctk turbo4 -ctv turbo4` or `-ctk turbo3 -ctv turbo3`.
2. Q4_K_M weights, unknown model? Start asymmetric: `-ctk q8_0 -ctv turbo4`.
3. Q4_K_M weights, Qwen2.5? Must use asymmetric. Symmetric is catastrophic.
4. Q4_K_M weights, Llama/Mistral/Cohere 24B+? Symmetric likely works, but validate PPL first.

---

## 6. Compression Analysis

Asymmetric K/V trades some K-side compression for quality safety. The V-side savings still dominate for long context:

| Config | K bpv | V bpv | KV compression vs fp16 | Quality |
|--------|-------|-------|:----------------------:|---------|
| q8_0/q8_0 | 8.5 | 8.5 | 1.9x | baseline |
| q8_0/turbo4 | 8.5 | 4.25 | 2.5x | +0.3-1.3% |
| q8_0/turbo3 | 8.5 | 3.125 | 2.8x | +1.1-2.1% |
| q8_0/turbo2 | 8.5 | 2.125 | 3.0x | +3.1-9.5% |
| turbo3/turbo3 | 3.125 | 3.125 | 5.1x | model-dependent |
| turbo4/turbo4 | 4.25 | 4.25 | 3.8x | +1.7-6.3% |

Even asymmetric q8_0/turbo3 achieves 2.8x total KV compression. At 128K context on a 70B model, that saves ~5GB of KV cache memory.

---

## 7. Independent Validation

These findings have been independently confirmed by multiple researchers:

**@sztlink (Felipe Sztutman)** — Qwen3-4B, RTX 4090, tonbistudio/turboquant-pytorch (2026-03-31):
- fp16-K + 2bit-V: 1.000 cosine similarity, 100% top-1 match
- All degradation from K compression, V has zero effect
- Boundary layer gap scales with K compression (-0.001 at 4-bit K, -0.010 at 2-bit K)

**@HyperionMS2040** — 10-model CUDA sweep, RTX 3090 (2026-03-31):
- Qwen2.5-7B: +3.6% PPL symmetric turbo3 vs +0.5% on Llama 3.1 8B
- Confirms model-family-dependent sensitivity pattern

**AMD HIP validation** — RX 9070 XT, gfx1201 (2026-03-29):
- Asymmetric q8_0/turbo4 confirmed: +1.0% PPL
- Symmetric catastrophic: PPL 81,277 (turbo3/turbo3 on Qwen2.5-7B Q4_K_M)
- Consistent with Metal and CUDA findings

---

## 8. Limitations

1. **Sensitivity root cause not proven.** We observe that Qwen2.5 is sensitive and Llama/Mistral are not, but we have not identified the specific architectural feature that causes this. Possible factors: attention head count, KV head ratio, weight distribution, or training procedure.

2. **PPL is a proxy.** Perplexity on wikitext-2 at 512 context is a noisy single metric. Task-specific benchmarks (reasoning, code, retrieval) may show different sensitivity patterns.

3. **Limited model families.** Tested on Qwen2.5, Llama-3.1, Mistral, Command-R+, phi-4, and Qwen3.5. Other architectures (DeepSeek, Gemma, GPT-class) are untested.

4. **Metal primary, CUDA/HIP secondary.** Most experiments run on Apple Silicon Metal. CUDA and HIP results confirm the pattern but with fewer model/config combinations tested.

---

## 9. Conclusion

V cache compression is effectively free. K cache precision is the dominant lever for attention quality. This is a direct consequence of the attention mechanism: K errors are amplified exponentially through softmax, while V errors scale linearly through the weighted sum.

The practical implication is simple: use asymmetric K/V configs. Compress V aggressively (turbo2 or turbo3), keep K at q8_0 or higher. This gives meaningful compression (2.5-3.0x total KV) with minimal quality loss (+0.3-2.1% PPL) on all tested models, including those where symmetric turbo fails catastrophically.

For models validated as symmetric-tolerant (Llama, Mistral, Cohere at 24B+), symmetric turbo3/turbo3 offers better compression (5.1x) at acceptable quality (+3.6-11.4% PPL). But asymmetric is always the safe default.

---

## 10. Prior Discussion

The asymmetric K/V findings were first shared publicly in the [llama.cpp TurboQuant discussion thread](https://github.com/ggml-org/llama.cpp/discussions/20969) starting March 26, 2026. Initial recommendations for asymmetric configs and the observation that K precision dominates quality were posted there before being formalized in this paper.

---

## References

- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- TurboQuant+ implementation: [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- llama.cpp fork: [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- Sparse V dequantization: [sparse-v-dequant.md](sparse-v-dequant.md)
- Boundary V: [layer-aware-v-compression.md](layer-aware-v-compression.md)
- M5 Max stress test: [m5-max-stress-test.md](m5-max-stress-test.md)
- Configuration recommendations: [turboquant-recommendations.md](../turboquant-recommendations.md)

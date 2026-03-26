# fp16 Centroid LUT (Vec Path) — Full Test Results

Branch: `experiment/decode-speed-parity` commit `e503d77`
Change: `constant half turbo_centroids_3bit_h[8]` replacing `constant float turbo_centroids_3bit[8]` in vec dequant only

## Quality

| Test | Main | Experiment | Delta |
|------|------|-----------|-------|
| PPL (8-chunk) | 6.211 | 6.211 | unchanged |
| PPL (32-chunk) | 5.471 | 5.471 | unchanged |

**Zero quality regression.**

## Prefill Speed

| Context | Main tok/s | Experiment tok/s | Delta |
|---------|-----------|-----------------|-------|
| 32-chunk | 2777 | 2784 | +0.3% |
| 2K | 4729 | 4694 | -0.7% |
| 4K | 3079 | 3062 | -0.6% |
| 8K | 2289 | 2261 | -1.2% |
| 16K | 1736 | 1736 | 0.0% |
| 32K | — | 1224 | — |

**No prefill regression.** All deltas within measurement noise.

## Decode Speed (THE WIN)

| Context | Main tok/s | Experiment tok/s | q8_0 tok/s | Main/q8_0 | Exp/q8_0 | Improvement |
|---------|-----------|-----------------|-----------|----------|---------|-------------|
| Short (~12) | 75.3 | **77.2** | 85.2 | 0.88x | **0.91x** | +2.5% |
| 8K | 59.2 | **67.3** | 77.7 | 0.76x | **0.87x** | +13.7% |
| 48K (PDF) | 36.7 | **39.0** | 55.6 | 0.66x | **0.70x** | +6.3% |

**Decode improvement at all context depths.** Biggest improvement at 8K (+13.7%). The half-precision LUT reduces constant cache pressure because half values are 2 bytes vs 4 bytes per entry.

## Change Description

One change to the Metal shader: the vec flash attention dequant (`dequantize_turbo3_0_t4`) uses `constant half[8]` instead of `constant float[8]` for the centroid table, and keeps the norm multiply in half precision (`xb->norm` is already stored as `ggml_half`).

The non-vec dequant (`dequantize_turbo3_0`) is unchanged — fp16 hurts the non-vec path due to half→float conversion overhead in the 16-element-per-call pattern.

## Variant A: Half LUT + Float Norm Broadcast (NEW BEST)

Further optimization: centroid lookup in half (2 bytes, small cache footprint), but norm as `float4(centroids) * float_norm` (single broadcast multiply instead of 4 half multiplies).

| Context | Main | fp16 LUT | **Half LUT + float norm** | q8_0 |
|---------|------|----------|--------------------------|------|
| Short | 75.3 | 77.2 | **78.7** | 85.2 |
| 8K | 59.2 | 67.3 | **68.8** | 77.7 |
| 48K PDF | 36.7 | 39.0 | **39.9** | 55.6 |
| vs q8_0 @ 48K | 0.66x | 0.70x | **0.72x** | — |

PPL: 6.211 (unchanged across all variants).

## All Decode Experiments Summary

| # | Approach | Short | 8K | 48K | vs main | Verdict |
|---|---------|-------|-----|-----|---------|---------|
| — | Main (float LUT) | 75.3 | 59.2 | 36.7 | baseline | — |
| — | No LUT (ceiling) | ~78 | 70.1 | — | theoretical max | — |
| 1 | fp16 LUT (vec only) | 77.2 | 67.3 | 39.0 | +6% | good |
| **2** | **Half LUT + float norm** | **78.7** | **68.8** | **39.9** | **+8.7%** | **BEST** |
| 3 | Register float4 select | — | — | — | -4% | worse |
| 4 | Inline switch | — | — | — | -13% | much worse |
| 5 | Split lo/hi + select | — | — | — | -10% | worse |
| 6 | Precomputed centroid*norm | — | — | — | +2% | marginal |
| 7 | fp16 LUT both paths | — | — | — | -10% | worse |
| 8 | Custom GGML_OP_TURBO_WHT | — | — | — | -7% | net negative |
| **9** | **cn[] precomputed registers** | **76.4** | **61.1** | — | **-10.5%** | **REVERT — register spill** |
| 8 | Custom GGML_OP_TURBO_WHT | — | — | — | -7% | net negative |

## Dead End: Precomputed cn[] Registers (Experiment 9)

Precomputed `cn[8] = {centroid[i]*norm}` into local float array in non-vec dequant. Helps prefill +3% but **destroys decode -10.5%**. The 8-float local array causes register spill that propagates through the Metal shader's flash attention template compilation. Even though only the non-vec function was changed, the vec kernel's register allocation is affected.

**DO NOT precompute centroid values into local arrays in the Metal shader.** Register pressure from even 8 floats causes cross-function spill in flash attention.

## Recommendation

**Main branch (half LUT + float norm broadcast) is optimal.** 9 experiments tested, 2 merged (#1 fp16 LUT, #2 float norm broadcast). The remaining 28% decode gap at 48K is irreducible within the current dequant function interface. Further improvement requires fused compressed attention (custom FA kernel).

## Experiments 10-11 (Additional Attempts)

### Experiment 10: Packed uint4 Register Centroids
Pack 8 half centroids into a uint4 (16 bytes, register-resident). Extract via `as_type<half2>(uint)[idx & 1]`.

| | Short | 8K | vs main |
|---|---|---|---|
| Main | 78.4 | 68.3 | baseline |
| Packed uint4 | 74.3 | 64.3 | **-5.5% — DEAD** |

The `as_type` extraction + uint4 data-dependent indexing is worse than constant half array. **Dead end.**

### Experiment 11: Fused FA Turbo3 Path (Post-Dot Norm)
Specialized `is_same<kd4_t, block_turbo3_0>` path in flash attention vec inner loop. Inlines the dequant, applies norm AFTER dot product instead of to each centroid element.

| | Short | 8K | vs main |
|---|---|---|---|
| Main | 78.4 | 68.3 | baseline |
| Fused FA | 78.4 | 68.9 | **+0% / +0.9% — MARGINAL** |

Codex predicted this: `float4 * scalar` norm is already 1 instruction. The compiler optimizes the dequant → dot pipeline. No gain from manual fusion.

**Both dead ends. 11 experiments total, 2 merged, 9 dead.**

# TurboQuant Decode Speed: Why Some Hardware Struggles

## The Problem

turbo3 decode speed varies dramatically across Apple Silicon generations:

| Hardware | Decode vs q8_0 (short) | Decode vs q8_0 (16K) | Has Tensor API |
|----------|----------------------|---------------------|---------------|
| M5 Max | 0.89x | ~0.82x | yes |
| M2 Pro | 0.73x | 0.67x (with 4-mag fix) | no |
| M1 Max | 0.84x | 0.45x (reported) | no |

M5 Max has near-parity. M2/M1 fall off a cliff at long context. This doc explains why and what we did about it.

## Root Cause: Constant Memory LUT Divergence

turbo3 dequant uses an 8-entry centroid lookup table (LUT) in Metal `constant` memory. Each element's value is determined by a 3-bit index (2-bit qs + 1-bit sign), so each thread in a 32-thread simdgroup may access a different LUT entry.

This "divergent access" pattern is where hardware generations differ:

- **M5 Max (Apple10)**: Efficient constant cache handles 8-way divergent access with minimal penalty. LUT costs ~14% of decode time.
- **M2 Pro (Apple8)**: Older constant cache serializes divergent access. LUT costs ~25% of decode time — **2x worse than M5**.
- **M1 Max (Apple7)**: Even worse, likely 30%+ LUT cost based on reported numbers.

## How We Know: Profiling Isolation

We added `TURBO_PROFILE_MODE` (0-4) to strip away dequant layers one at a time:

| Mode | What | M5 (% of ceiling) | M2 (% of ceiling) |
|------|------|-------------------|-------------------|
| 1 | No dequant at all | 78.9 (100%) | 24.5 (100%) |
| 2 | + read norm only | 75.1 (95%) | 22.1 (90%) |
| 4 | + read all bytes | 75.2 (95%) | 21.9 (89%) |
| 3 | + qs extraction + LUT | 64.9 (82%) | 16.4 (67%) |
| 0 | + signs + full LUT | 59.2 (75%) | 14.0 (57%) |

**Key finding: No-dequant turbo3 is 12% FASTER than q8_0 on M2 Pro** (24.5 vs 21.9) because the compressed cache moves less data over the memory bus. The compression FORMAT is not the problem. The dequant COMPUTATION is.

**The LUT accounts for:**
- M5: 13.7% of ceiling (Mode 4 → Mode 3)
- M2: 25.1% of ceiling — **2x worse**

Reading the bytes (norm + qs + signs) costs ~10% on both chips. That's not the bottleneck.

## What We Tried (7 Approaches)

| # | Approach | M2 8K tok/s | vs Main | Why it works/doesn't |
|---|----------|-------------|---------|---------------------|
| 1 | **4-mag LUT + XOR sign** | **15.1** | **+38%** | **Halves constant addresses (4 vs 8). Sweet spot on M2.** |
| 2 | Batched byte extract | 13.7 | +25% | Better byte reading, still 8 LUT addresses |
| 3 | Deferred norm multiply | 12.9 | +18% | Loses ILP — per-element norm hides LUT latency |
| 4 | 2-pair half2 LUT | 12.0 | +10% | Ternary overhead exceeds savings from 2 addresses |
| 5 | Select chain (zero LUT) | 11.9 | +9% | Too much ALU — branches on M2 GPU |
| 6 | Bit-arithmetic | 11.6 | +6% | Pure ALU, zero memory — but ALU cost too high |
| 7 | Non-vec FA (nl=2) | 10.2 | -7% | Non-vec kernel not designed for single-token decode |

Also tested on M5 Max (no help needed):
- float cn[8] registers: 75.2 (spills to stack on Metal)
- half cn[8] registers: 74.4 (also spills)
- Split 2×4 half LUT: 74.0 (branch overhead)

## The Fix: Auto-Detection

The build auto-detects hardware at Metal library compile time:

```
M1/M2/M3/M4 (has_tensor=false) → TURBO_USE_4MAG=1 → 4-entry magnitude LUT
M5+          (has_tensor=true)  → TURBO_USE_4MAG=0 → 8-entry full LUT
```

Each chip gets its optimal dequant path. No user configuration needed.

### How 4-mag LUT works

The 8 centroids have structure: 4 magnitudes × 2 signs. We split the 3-bit index:
- **2-bit qs** → selects magnitude from 4-entry LUT (4 possible constant addresses)
- **1-bit sign** → reverses magnitude index via XOR, then negates via ALU multiply
- **Norm** → multiplied per-element (provides ALU work that hides constant memory latency)

The XOR trick: for negative values (sign=0), the magnitude index is reversed (qs=0 → highest magnitude). `qs ^ 0x3` flips the 2-bit index without a branch.

## Why CUDA Doesn't Have This Problem

@spiritbuun's CUDA fork achieves 97.5% of q8_0 decode. The key difference:

- **CUDA has 255 registers per thread** — cn[8] fits in registers without spilling
- **Metal has a smaller register file** — cn[8] spills to thread stack memory
- **CUDA warp semantics** — better divergent access handling across 32 threads
- **Metal simdgroup semantics** — constant cache serializes more on older chips

The register LUT approach works perfectly on CUDA but is fundamentally incompatible with Metal's register file on current hardware.

## Remaining Gap

Even with 4-mag LUT, M2 Pro decode at 8K is 15.1 vs 24.5 ceiling (62%). The remaining 38% gap needs kernel-level changes:

1. **SMEM pre-dequant**: Pre-dequant entire KV blocks into threadgroup memory before dot products. Would eliminate constant memory from the inner loop entirely. Requires modifying the FA vec kernel template.

2. **Per-group device-memory cached centroid×norm**: Store 8 pre-computed centroid×norm values per 128-element group alongside the block data. Dequant reads from device memory (sequential) instead of constant memory (divergent). Changes the block format.

3. **Fused Q·centroid attention**: Compute attention scores directly from quantized indices without full dequantization. Precompute Q·centroid table (8 values per block), then each K element is a table lookup. Complex (custom FA kernel).

These are tracked in GitHub issue #39.

## Summary

| What | Result |
|------|--------|
| Root cause | Constant memory LUT divergence, 2x worse on M2 vs M5 |
| Best fix found | 4-mag LUT (+38-45% on M2, auto-detected) |
| M5 impact | Zero regression (uses separate code path) |
| Remaining gap | 38% to ceiling, needs kernel-level surgery |
| CUDA comparison | buun gets 97.5% via register LUT (not portable to Metal) |

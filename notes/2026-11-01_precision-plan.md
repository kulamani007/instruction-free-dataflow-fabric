# Precision plan: INT8 default, optional 4-bit per layer

**Author:** Kulamani Rout · **First recorded:** 1 November 2026 · **Status:** agreed as working plan; **to be discussed further** (choice of 4-bit format open)

**Code:** [`model/precision_compare.py`](../model/precision_compare.py)

## Decision

- **INT8 is the default precision of every engine.** INT8 post-training quantisation of
  CNNs is close to lossless, the tools (TFLite, ONNX) support it, and the baselines this work
  compares against (Coral Edge TPU, Gemmini) are INT8.
- **A 4-bit mode is optional, chosen by the compiler per layer.** Layers that tolerate it run
  in 4-bit; sensitive layers (typically the first and last layers and depthwise layers)
  stay INT8. 4-bit halves weight memory for those layers, so more models fit as circuits.
- **Not FP4 only.** FP4 is *less* precise than INT8 (measured below); FP4-only would cost
  accuracy and comparability with no benefit at edge scale that the 4-bit mode does not
  already give.
- **NVFP4 and other NVIDIA-proprietary formats are ruled out** and will not be considered.
  Open standards stay eligible even when NVIDIA is one of their co-developers.
- **4-bit candidates: MXFP4 (OCP open standard) and INT4 with block scales.** Both are kept
  as options (decided 1 November 2026); the choice between them is made by experiment.

## Why INT8 beats FP4 on precision

FP4 (E2M1) has 15 distinct values (±0.5, 1, 1.5, 2, 3, 4, 6 and zero); INT8 has 256.
Rounding error on synthetic weight-like data (SQNR in dB; higher is more precise; +6 dB is
about one bit):

| Format | Gaussian | Laplacian | Gaussian + rare outliers |
|---|---|---|---|
| **INT8**, one scale per tensor | **39.8** | **35.3** | **26.1** |
| INT4, one scale per tensor | 14.6 | 10.4 | 1.8 |
| FP4, one scale per tensor | 17.9 | 14.5 | 5.6 |
| INT4, one scale per 32 values | 20.2 | 18.3 | 18.9 |
| MXFP4 (OCP), one power-of-two scale per 32 values | 18.8 | 17.9 | 18.4 |

INT8 is more precise than every 4-bit option in every case: by 17–20 dB (about 3 bits) on
Gaussian and Laplacian data, but only by about 7 dB when rare outliers stretch INT8's single
per-tensor scale (per-channel INT8 scales would recover most of that; not yet measured).
*Corrected 2 Nov 2026: this sentence first said "about 19–21 dB more precise than any 4-bit
option", which holds only for the Gaussian case.* FP4 beats INT4 only
with a single scale per tensor, because its uneven spacing handles large values better.
With a scale per block of 32 values, INT4 and MXFP4 are about equal. FP4 is popular for
large language models because it halves memory traffic where weight bandwidth dominates,
not because it is more precise.

Limits: synthetic data; real accuracy needs trained, quantised models (with
quantisation-aware training at 4-bit).

## Link to the lookup-table idea

A 4-bit × 4-bit product table has 256 entries (about 256 bytes). Every FP4 × FP4 product,
multiplied by 4, is an integer in [−144, 144] (checked in the script), so it fits a 9-bit
integer exactly: products can be looked up and **accumulated with plain integer adders**, with
each block's scale applied once at the end. A 4-bit engine can therefore be "lookup + add, no
multiplier", the practical form of the idea in
[2026-11-01_lut-multiplication-idea.md](2026-11-01_lut-multiplication-idea.md).

## Open points for discussion

1. **Which 4-bit format: INT4 with block scales, or MXFP4?** They measured about equal here;
   decide on accuracy and hardware cost in a real-model experiment. MX (including MXFP4) is
   an open standard published through the Open Compute Project and developed jointly by
   Microsoft, AMD, Intel, Meta, NVIDIA and Qualcomm, so it is allowed under the rule above
   (only NVIDIA-proprietary formats such as NVFP4 are excluded).
2. Which layers may drop to 4-bit, and who decides: a sensitivity test per layer in the
   compiler, or quantisation-aware training per model?
3. Block size (16 / 32 / 64 values per scale) versus the window unit's fetch width.
4. Accuracy experiment on real models (e.g. the tiny-skeleton models and MobileNetV2).

## Reading

- B. D. Rouhani et al., "Microscaling Data Formats for Deep Learning," arXiv:2310.10537, 2023.
  <https://arxiv.org/abs/2310.10537> (the founding MX paper; defines MXFP4 and evaluates MX
  formats). A copy is kept locally in `papers/`, which is not committed: the paper's arXiv
  licence does not allow redistribution.
- OCP Microscaling Formats (MX) v1.0 Specification (formal definition of the formats).

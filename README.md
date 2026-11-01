# Instruction-Free Dataflow Fabric

**Deterministic, instruction-free multi-model inference on a configurable spatial dataflow fabric for real-time edge systems**

Author: **Kulamani Rout** · [github.com/kulamani007](https://github.com/kulamani007)

## Core idea

The architecture keeps any general-purpose instruction processor out of the inference loop. A compiler
transforms the model graph into hardware configuration, routing, memory, and schedule metadata.
Dedicated dataflow stages in fixed silicon then execute inference directly on streaming inputs.
Because the whole schedule is known before inference begins, the compiler can bound each model's
worst-case latency, even when several models share the fabric, while the fabric stays reprogrammable
for new models.

## Contents

| Path | What it is |
|---|---|
| [`proposal/`](proposal/) | Research proposal (PDF + Markdown source) |
| [`model/window_bank_model.py`](model/window_bank_model.py) | Python functional model: stride-driven address generator, bank-partitioned single-cycle window fetch, FULL vs SHIFT fetch modes, SRAM-read counts on MobileNetV1 |
| [`model/composability_model.py`](model/composability_model.py) | Event-driven model: latency of co-running models on a time-shared NPU vs a statically partitioned fabric |
| [`model/fit_map.py`](model/fit_map.py) | Skeleton fit map: which CNNs a fixed skeleton can become (circuit / folded / streamed), and at what cost |
| [`model/tiny_skeleton.py`](model/tiny_skeleton.py) | Tiny skeleton v0: three custom 5–8-layer CNNs resident at once, sized against the ZU19EG FPGA |
| [`model/precision_compare.py`](model/precision_compare.py) | Rounding error of INT8, INT4, FP4 and MXFP4 on weight-like data |
| [`model/workloads/`](model/workloads/) | Exact layer tables of 11 reference CNNs, with the extraction script |
| [`figures/`](figures/) | Figures produced by the models |
| [`RUNBOOK.md`](RUNBOOK.md) | Every run and test, dated, with exact commands and results; setup and regression check |
| [`tools/`](tools/) | `check_all.sh` regression check; one-off proposal PDF patch script |
| [`notes/`](notes/) | Dated design notes and results; start with [core message and claims](notes/2026-11-01_core-message-and-claims.md) |

## Idea log

| Date | Entry |
|---|---|
| Aug–Sep 2026 | Concept development: processor-free, layer-mapped spatial fabric for edge AI |
| 22 Sep 2026 | Proposal v1: configurable stage fabric; compiler-generated configuration, routing, memory, and schedule; no instruction processor in the inference loop |
| 22 Sep 2026 | Window-extraction unit: cyclic 2-D RAM banking for single-cycle M×N window fetch; multi-hot bank-enable mask for clock gating |
| 23 Sep 2026 | Stride-driven window address generator; FULL vs SHIFT fetch modes; functional model and MobileNetV1 read analysis (57% fewer window-fetch SRAM reads) |
| 24 Sep 2026 | [Heterogeneous convolution engines per stage with multi-model co-scheduling and provable non-interference](notes/2026-09-24_heterogeneous-stage-engines.md) |
| 29 Sep 2026 | Validation method: FPGA as cycle-accurate emulation, ASIC flow for area/timing/energy, MPW test chip as stretch goal |
| 30 Sep 2026 | [External memory tier with double buffering; polyphase (space-to-depth) instead of im2col for strided layers](notes/2026-09-30_external-memory-and-polyphase.md) |
| 1 Nov 2026 | [Core message: a CNN skeleton that becomes any model in its range, as a circuit; testable claims C1–C4](notes/2026-11-01_core-message-and-claims.md) |
| 1 Nov 2026 | [Composability experiment (time-shared NPU inflates small models' worst case up to 249×) and skeleton fit map of 11 CNNs](notes/2026-11-01_composability-and-fit-map-results.md) |
| 1 Nov 2026 | [Lookup tables instead of multipliers: assessed (INT8 no; ≤ 4-bit and neuron-as-table yes)](notes/2026-11-01_lut-multiplication-idea.md) |
| 1 Nov 2026 | [Tiny skeleton v0: three custom CNNs resident at once, first FPGA prototype target](notes/2026-11-01_tiny-skeleton-v0.md) |
| 1 Nov 2026 | [Precision plan: INT8 default, optional 4-bit per layer (MXFP4 or block INT4); NVFP4 ruled out](notes/2026-11-01_precision-plan.md) |

## Running the models

Full setup, dated run history and the regression check are in [`RUNBOOK.md`](RUNBOOK.md). Quick start:

```bash
pip install numpy matplotlib
python model/window_bank_model.py
python model/composability_model.py && python model/plot_composability.py
python model/fit_map.py && python model/plot_fit_map.py
python model/tiny_skeleton.py
python model/precision_compare.py
```

Re-extracting the layer tables in `model/workloads/` also needs `torch`, `torchvision` and `ultralytics`.

## Citing this work

If you use or build on these ideas, please cite this repository (see `CITATION.cff`) and contact the author.

## License

Documents and design notes: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse requires attribution.
Code: MIT. See `LICENSE`.

# Instruction-Free Dataflow Fabric

**Deterministic, instruction-free multi-model inference on a configurable spatial dataflow fabric for real-time edge systems**

Author: **Kulamani Rout** (B.Tech EEE, IIIT Bhubaneswar) · [github.com/kulamani007](https://github.com/kulamani007)

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
| [`notes/`](notes/) | Dated design notes for ideas under development |

## Idea log

| Date | Entry |
|---|---|
| Aug–Sep 2026 | Concept development: processor-free, layer-mapped spatial fabric for edge AI |
| 22 Sep 2026 | Proposal v1: configurable stage fabric; compiler-generated configuration, routing, memory, and schedule; no instruction processor in the inference loop |
| 22 Sep 2026 | Window-extraction unit: cyclic 2-D RAM banking for single-cycle M×N window fetch; multi-hot bank-enable mask for clock gating |
| 23 Sep 2026 | Stride-driven window address generator; FULL vs SHIFT fetch modes; functional model and MobileNetV1 read analysis (57% fewer window-fetch SRAM reads) |
| 24 Sep 2026 | [Heterogeneous convolution engines per stage with multi-model co-scheduling and provable non-interference](notes/2026-09-24_heterogeneous-stage-engines.md) |

## Running the model

```bash
pip install numpy
python model/window_bank_model.py
```

## Citing this work

If you use or build on these ideas, please cite this repository (see `CITATION.cff`) and contact the author.

## License

Documents and design notes: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reuse requires attribution.
Code: MIT. See `LICENSE`.

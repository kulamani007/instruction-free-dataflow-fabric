# Core message and research claims

**Author:** Kulamani Rout · **First recorded:** 1 November 2026 · **Status:** agreed direction for pitching; the proposal text has not been updated to match yet

## The idea, in the author's own words

Software runs on processors because it is turned into instructions. A CNN does not have
to be: it can be built directly as hardware, where data goes in and the result comes out,
with no instructions fetched or decoded. Today that only happens by designing a custom
chip or FPGA bitstream for one model.

The idea is a **fixed hardware skeleton** that any CNN *within its range* can be fitted
into. The compiler does not produce a program; it configures the skeleton so that it
*becomes* that model's circuit. Load different settings and it becomes a different model.
No processor in the loop. Scope: convolutional models (transformers later, they are harder).

## What listeners should hear

> "Today a neural network runs as a program on a processor. I want it to run as a
> circuit, on one reusable chip, without redesigning the chip for each model."

Three consequences follow from the skeleton, and they are what make it research rather
than a slogan:

1. **No instruction overhead:** nothing is fetched or decoded during inference.
2. **Timing is known in advance:** a circuit does not make runtime decisions.
3. **Several models can occupy different parts of the skeleton without disturbing each
   other** (composability). See the experiment in
   [2026-11-01_composability-and-fit-map-results.md](2026-11-01_composability-and-fit-map-results.md).

## Questions to be ready for

- **"Isn't configuration just instructions?"** No: instructions are control decisions made
  every cycle during the run; configuration is decided once, before the run, and stays fixed.
  The textbook terms are **spatial computing** (a circuit) versus **temporal computing**
  (a program).
- **"What is the range of a skeleton?"** Not one number. Parameter count matters most for
  whether a model fits as a circuit (weights must sit on-chip next to the MACs), but realistic
  capacities are megabytes, not billions: MobileNetV1 has 4.2 M parameters, ResNet-50 25.6 M,
  YOLOv8n 3.2 M, and even the largest CNNs (ConvNeXt-XL) are around 350 M. The range is
  several dimensions: engine count, MACs per engine, on-chip weight SRAM, activation buffer,
  kernel sizes and strides, and the operator set. Past the range a model does not fail; it
  folds (engines shared over time) or streams weights from DRAM, and only the cost changes.
- **"Why CNNs when everything is transformers?"** Real-time edge perception (detection,
  keyword spotting, segmentation on drones and sensors) is still mostly CNNs or CNN hybrids;
  transformer blocks are planned for year 4.
- **"How is this different from X?"** See below.

## Closest prior work (must be cited and distinguished)

| Work | What it already does | What it does not do |
|---|---|---|
| Groq TSP (Abts et al., ISCA 2020) | Deterministic execution; compiler schedules everything statically | Still instruction-driven; datacenter; no multi-model timing guarantees |
| Hailo-8 (commercial edge NPU) | Compiler maps layers onto distributed compute, memory and control blocks in fixed silicon | No published timing guarantees for concurrent models |
| Plasticine / SambaNova, CGRAs | Configuration-driven spatial dataflow, reprogrammable | Not edge real-time; no per-model latency bounds |
| FINN, hls4ml, fpgaConvNet | CNN as a circuit, no processor | New bitstream per model |
| DLA (Abdelfattah et al., FPL 2018) | FPGA overlay, many models on one bitstream | Instruction stream dispatches each layer |
| Restuccia & Biondi (RTSS 2021); DART (RTSS 2019); NeuOS (ATC 2020) | Response-time analysis / real-time scheduling of DNNs | Analysis or scheduling on top of hardware that is not predictable |
| MoCA (HPCA 2023), Planaria (MICRO 2020), Zeng et al. (TC 2023) | Multi-tenant DNN accelerators; Zeng reserves DRAM bandwidth via a hypervisor | Reduce interference, do not bound it at compile time |

Groq matters most: it shows determinism does not require removing instructions. The
problem is **dynamic arbitration and shared-memory contention**, not instruction fetch
itself. The proposal's section 1 motivation should be phrased that way.

The open gap (from quick searches, not yet a full review): **a formal worst-case latency
bound for several CNNs sharing one reprogrammable chip, including DRAM, provided by the
hardware and compiler themselves.**

## Claims (each testable, each could turn out false)

| # | Claim | Test | Status |
|---|---|---|---|
| C1 Composability | A model's latency is cycle-identical alone or co-running | FPGA, against Gemmini with shared DRAM | Motivated by simulation (baselines inflate 10–249×); hardware proof pending |
| C2 Tight bounds | Compile-time worst-case bound, including DRAM, within a small margin of measured | Bound vs measured across workload mixes | Not tested; needs a DRAM timing model (Ramulator / DRAMSys) |
| C3 Price of determinism | Isolation costs at most X % throughput vs an equal-resource, work-conserving baseline | Equal-resource comparison | First numbers: largest model ~2.9× its full-chip latency; X to be set from data |
| C4 Compiler decides exactly | Joint placement + DRAM-slot allocation admits model sets that static 50/50 rejects, with no false accepts | Random and real workload sets | Objective choice shown to matter a lot; exact admission test not built |

"Removing dispatch saves control energy" is kept as a secondary result, not a headline:
Groq and modern NPUs already keep dispatch overhead low.

## Who to approach

Labs at the intersection of computer architecture and real-time systems, for example the
Biondi group (Scuola Sant'Anna, Pisa), Kees Goossens (TU Eindhoven, composable SoCs),
Hyoseung Kim (UC Riverside, DART), Sophia Shao (UC Berkeley, Gemmini, MoCA). Venues:
RTSS / RTAS / EMSOFT and DAC / ISCA / MICRO.

## Sources consulted

- Groq TSP, ISCA 2020: <https://fuse.wikichip.org/?p=3005>
- Hailo-8 layer mapping: <https://spectrum.ieee.org/israeli-edgeai-startup-gets-60million-preps-for-mass-production>, <https://www.vision-systems.com/boards-software/article/14169509/hailo-8-deep-learning-application-processor>
- Restuccia & Biondi, RTSS 2021: <https://www.iris.sssup.it/retrieve/9e4e6f2a-7d24-4de6-b45f-cc2908ce2a20/RTSS21.pdf>
- DART, RTSS 2019: <https://intra.ece.ucr.edu/~hyoseung/pdf/rtss19-dart.pdf>
- NeuOS, ATC 2020: <https://www.usenix.org/system/files/atc20-bateni.pdf>
- MoCA, HPCA 2023: <https://people.eecs.berkeley.edu/~ysshao/assets/papers/moca-hpca2023.pdf>
- Planaria, MICRO 2020: <https://cseweb.ucsd.edu/~hadi/doc/paper/2020-micro-planaria.pdf>
- Zeng et al.: <https://nicsefc.ee.tsinghua.edu.cn//nics_file/pdf/bb2bf540-64c7-4ab4-984b-3528921bd277.pdf>
- Composable many-core timing analysis: <https://arxiv.org/pdf/1905.13503>

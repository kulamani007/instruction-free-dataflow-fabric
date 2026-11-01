# Results: composability experiment and skeleton fit map

**Author:** Kulamani Rout · **First recorded:** 1 November 2026 · **Status:** first simulation results (Python models); not yet hardware-validated

Two Python models support the claims in
[2026-11-01_core-message-and-claims.md](2026-11-01_core-message-and-claims.md).
Both are simulations with stated simplifications; the negative findings are kept on
purpose, because they define the research questions.

## 1. Composability: does co-running change a model's latency?

**Code:** [`model/composability_model.py`](../model/composability_model.py) ·
**Figure:** [`figures/composability.png`](../figures/composability.png)

**Setup.** Same hardware for every system: 256 INT8 MACs at 500 MHz, 3.2 GB/s DRAM. Real
layer shapes (MobileNetV1 568.7 MMAC, DS-CNN 2.7 MMAC, KWT-1 73.7 MMAC). Each layer moves
its INT8 weights plus input and output activations through DRAM, and runs at min(compute
rate, bandwidth rate). Models run periodically with random phases (40 runs × 1 s).

| System | How resources are shared |
|---|---|
| B1 time-shared NPU | Whole array runs one layer at a time (FIFO or rate-monotonic priority) |
| B2 split compute | Each model owns a MAC slice; DRAM bandwidth shared, work-conserving |
| P proposed | Same MAC slices + a fixed DRAM bandwidth share (TDM) per model, chosen at compile time |

**Scenario A** (MobileNetV1 at 30 FPS, DS-CNN every 40 ms, KWT-1 at 10 Hz), worst-case latency:

| Model | Alone, full chip | B1 priority | B2 | P (fair objective) |
|---|---|---|---|---|
| MobileNetV1 | 6.46 ms | 6.47 ms | 12.3 ms | **18.7 ms** (fixed) |
| DS-CNN | 0.050 ms | **6.40 ms (129×)** | 0.150 ms | 0.141 ms (fixed) |
| KWT-1 | 0.74 ms | **7.25 ms (9.8×)** | 1.72 ms | 2.15 ms (fixed) |

**Scenario B** (two MobileNetV1 cameras + DS-CNN, DRAM-heavy): DS-CNN worst case reaches
**12.4 ms (249×)** on B1; the second camera model reaches 2.0× under priority scheduling.

**Findings.**
1. **Interference on a time-shared NPU is large and comes from the co-runners.** A small
   model's worst case is set by the largest model it shares with, not by itself.
2. **Static partitioning gives each model one fixed latency**, equal to its compile-time
   bound, alone or co-running. The keyword spotter's guaranteed 0.14 ms is ~45× better than
   its time-shared worst case.
3. **Isolation has a price (claim C3).** The largest model's guaranteed latency is ~2.9× its
   full-chip latency (still meets 30 FPS). Comparing B2 and P isolates the DRAM policy:
   MobileNetV1 worst case 12.3 ms with shared DRAM vs 18.7 ms with fixed shares, because
   fixed shares cannot borrow bandwidth other models leave idle. Open question: keep the
   bound but let a model use idle bandwidth (*predictable* but not *composable*, the
   distinction used in the CompSOC line of work).
4. **The compiler's objective decides who pays (claim C4).** Optimising only for deadlines
   gave KWT-1 1/32 of the array: guaranteed 18.4 ms, worse than its time-shared worst case.
   Spreading the slowdown evenly gave 2.15 ms on the same hardware.

**Limits.** P's isolation holds by construction in this model (its rates depend only on its
own shares); the experiment measures the size of interference and the cost of isolation,
not that hardware provides isolation. DRAM is an ideal fluid pipe (no refresh, row
conflicts): B2's 1.03–1.57× DRAM interference is therefore a lower bound, and C2 (bound
tightness) is not tested. All operators run at 100 % array utilisation; baselines have zero
dispatch overhead (favours them).

## 2. Skeleton fit map: which CNNs can a fixed skeleton become?

**Code:** [`model/fit_map.py`](../model/fit_map.py), layer tables in
[`model/workloads/`](../model/workloads/) · **Figure:** [`figures/fit_map.png`](../figures/fit_map.png)

**Setup.** A skeleton is E identical engines, each with a MAC array and its own weight
SRAM. Three sizes: Small (16 engines, 2 MB, 2 TOPS), Medium (64 engines, 8 MB, 8 TOPS),
Large (128 engines, 32 MB, 33 TOPS). A toy compiler splits the layer sequence into
contiguous groups and gives each group one or more engines, maximising throughput.
Eleven CNNs, with exact layer tables extracted from torchvision / Ultralytics (parameter
and MAC totals match the published figures, e.g. ResNet-50 25.56 M / 4.09 GMAC, YOLOv8n
3.16 M / 4.37 GMAC).

Fit classes, all processor-free; only the cost differs:
- **Circuit:** every layer has its own engine(s); weights stay on-chip.
- **Folded:** weights on-chip, but engines serve several layers on a fixed schedule.
- **Streams weights:** weights re-read from DRAM every frame.

| Model | Weights | Small | Medium | Large | Operators beyond proposal op set |
|---|---|---|---|---|---|
| DS-CNN | 0.02 MB | Circuit | Circuit | Circuit | kernel 10×4 > 5×5 |
| SqueezeNet1.1 | 1.24 MB | Folded | Circuit | Circuit | concat |
| MobileNetV3-Small | 2.53 MB | Streams | Folded | Circuit | channel scale (SE) |
| YOLOv8n | 3.15 MB | Streams | Folded | Circuit | concat, split, upsample |
| MobileNetV2 | 3.47 MB | Streams | Folded | Circuit | none |
| MobileNetV1 | 4.21 MB | Streams | Circuit | Circuit | none |
| EfficientNet-B0 | 5.25 MB | Streams | Folded | Circuit | channel scale (SE) |
| YOLOv8s | 11.1 MB | Streams | Streams | Circuit | concat, split, upsample |
| ResNet-18 | 11.7 MB | Streams | Streams | Circuit | kernel 7×7 > 5×5 |
| ResNet-50 | 25.5 MB | Streams | Streams | Circuit | kernel 7×7 > 5×5 |
| VGG-16 | 138 MB | Streams | Streams | Streams | none |

**Findings.**
1. **On-chip weight capacity decides whether a model becomes a circuit.** The "parameter
   capacity" intuition is right; the realistic numbers are megabytes. ResNet-18 is a circuit
   only on the 32 MB skeleton, even though the 8 MB one has enough compute for it.
2. **Layer count matters too.** MobileNetV2 has fewer weights than MobileNetV1 but 53 layers
   vs 28, so on Medium it folds while MobileNetV1 is a circuit.
3. **A pure circuit leaves engines idle.** One layer per engine wastes MACs because layer
   work differs by ~100×: MobileNetV1 on Medium uses 52 % of the MACs, ResNet-50 on Large
   28 % (its weights force many engines). Folding could recover up to 1.2–3.6× throughput,
   but that number is idealised (switching engines between layers costs nothing here).
   Circuit vs fold is a real compiler trade-off (RQ3).
4. **The proposal's operator set is too small for modern CNNs.** Detection (YOLOv8) needs
   concat, split and upsample; MobileNetV3 / EfficientNet need channel scaling
   (squeeze-excite); ResNet's 7×7 stem and DS-CNN's 10×4 kernel exceed a 5×5 window unit
   (decompose or enlarge). Section 3.1 of the proposal should list these.

**Limits.** Activation memory is not modelled: line buffers, and more importantly the buffers
that hold a skip connection until it is merged (YOLO's long skips between backbone and head
may need whole feature maps). Per-operator array utilisation is assumed 100 %; a layer can
be split across any number of engines; reconfiguration time in folded mode is zero. Frame
rates printed by the model are upper bounds.

## Next steps (rough priority)

1. Skip-connection buffer sizing in the fit map (likely the real limit for detection models).
2. A third static option in the composability model: a time-triggered schedule that gives
   each model the whole array in fixed windows (deterministic, may avoid the 2.9× cost).
3. Work-conserving DRAM with a guaranteed minimum share; then a DRAM timing model for C2.
4. Per-operator utilisation (depthwise on a dense array) in both models.
5. Update the proposal (section 1 motivation, section 3.1 op set, section 6 with Groq and
   Hailo, claims C1–C4) once the claims are agreed.

## Reproducing

```bash
pip install numpy matplotlib
python model/composability_model.py
python model/plot_composability.py
python model/fit_map.py
python model/plot_fit_map.py
```

Re-extracting the layer tables additionally needs `pip install torch torchvision ultralytics`,
then `python model/workloads/extract_layers.py`.

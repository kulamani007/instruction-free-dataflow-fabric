# Tiny skeleton v0: three custom CNNs resident at once (first prototype target)

**Author:** Kulamani Rout · **First recorded:** 1 November 2026 · **Status:** sized in Python; models untrained; candidate target for the first FPGA prototype

**Code:** [`model/tiny_skeleton.py`](../model/tiny_skeleton.py)

## The idea

Design three small custom CNNs (5–8 layers each) for three different edge tasks, and build
a skeleton just big enough that **all three fit as circuits at the same time**: every layer
of every model has its own engine, all weights stay on-chip, and nothing is shared between
the models. This is the smallest complete demonstration of the whole concept: a model
becomes a circuit, several models live side by side, and reloading the configuration turns
the skeleton into different models.

## The three models

| Model | Task (style) | Input | Layers | Compute | Weights | Rate |
|---|---|---|---|---|---|---|
| VisionNet | Image classification (CIFAR-10) | 32×32×3 | 5 conv + FC | 7.52 MMAC | 70.2 KB | 30 fps |
| KWSNet | Keyword spotting (Speech Commands) | 49×10 MFCC | conv, 2× (dw + pw), FC | 0.92 MMAC | 8.1 KB | every 20 ms |
| PersonNet | Person detection (visual wake words) | 96×96×1 | conv, 3× (dw + pw), FC | 1.35 MMAC | 3.4 KB | 15 fps |

All use only operators in the proposal's set (conv, depthwise, pointwise, global pool + FC,
ReLU) and kernels ≤ 3×3, so the 5×5 window unit covers them.

## Skeleton v0

- **20 identical engines** (one per layer), each with M INT8 MACs, weight SRAM, a line buffer
  (≤ 4 KB needed) and the window unit.
- **Fits the ZU19EG easily.** At M = 16: 320 DSPs (16 % of 1,968) and ~1.36 MB of block RAM
  (32 % of 4.2 MB), with UltraRAM unused. Even M = 4 meets every deadline (VisionNet
  ≤ 9.4 ms against 33 ms), using 80 DSPs.

| MACs per engine | DSPs | VisionNet latency | KWSNet | PersonNet |
|---|---|---|---|---|
| 4 | 80 | 9.41 ms | 1.15 ms | 1.69 ms |
| 16 | 320 | 2.35 ms | 0.29 ms | 0.42 ms |
| 64 | 1,280 | 0.59 ms | 0.07 ms | 0.11 ms |

(Latency = sum of layer times at 200 MHz with no overlap between layers; line-buffer
streaming would overlap layers, so these are upper bounds.)

## Findings

1. **Identical engines waste weight memory.** Sizing every engine for the largest layer
   (36 KB → 64 KB) leaves 94 % of the weight SRAM empty, because layer weights differ by
   about 100×. Letting a big layer span several engines fixes most of it: with 8 KB per
   engine the skeleton needs 27 engines instead of 20 but only 216 KB of weight SRAM instead
   of 1,280 KB (6× less). Engine granularity is a real design parameter.
2. **At this size, the composability benefit is small, and the honest comparison says so.**
   A time-shared NPU with the same 320 MACs runs each model 20× faster (it puts all MACs on
   every layer) and its worst-case inflation from co-running is only 1.1–1.8×, because the
   models are tiny and load the hardware under 1 %. The large interference seen earlier
   (129–249×) appears when models are big relative to the hardware. So this prototype
   demonstrates the **mechanism** (circuits, side by side, reprogrammable, cycle-identical
   latency), not an advantage over NPUs. The advantage must be shown at a scale where
   models compete for resources.
3. **Custom models are good for the prototype, weak as evidence.** Because they are
   designed for the skeleton, reviewers will discount them. Pair them with standard
   benchmarks (MLPerf Tiny: DS-CNN, ResNet-8, MobileNetV1-0.25 for visual wake words) when
   making claims, and train the custom models to report accuracy.

## Next steps

1. Train the three models (CIFAR-10, Speech Commands, visual wake words) and quantise to
   INT8, to have real accuracy numbers.
2. Pick engine granularity (MACs and SRAM per engine) with the multi-engine layer split.
3. RTL for one engine (window unit + MAC array + weight SRAM), verified against the Python
   model; then 20–27 engines on the ZU19EG and measure that each model's latency is
   identical alone and co-running (claim C1 on hardware).
4. Scale up the workload (or scale down the skeleton) until models compete for resources,
   to show where composability pays off.

# Design note: heterogeneous convolution engines per stage, with multi-model co-scheduling

**Author:** Kulamani Rout · **First recorded:** 24 September 2026 · **Status:** idea, not yet modelled

## Problem

Every stage of the fabric is fixed in silicon, but models mix convolution types:
standard convolution, depthwise convolution, and pointwise (1×1) convolution, in a
different order in every model. A stage must be able to host whichever type the
compiler assigns to it.

## Original idea

Give every stage all three convolution types as sub-units
(standard | depthwise | pointwise). The compiler maps each model layer to the matching
sub-unit of the stage it is placed on. Sub-units left idle by one model can be used by
another model concurrently, so several models can share the fabric without taking over
each other's paths.

## Refinement

1. **Two engines, not three.** Pointwise convolution is standard convolution with a 1×1
   kernel; both reduce to matrix multiplication and share one *dense engine*. Depthwise
   convolution has almost no cross-channel reuse and runs poorly on a large dense array,
   so it gets a small dedicated *depthwise engine*.
2. **Size by workload.** In MobileNetV1, 1×1 convolution is about 95% of the MACs and
   depthwise about 3% (Howard et al., 2017). Equal-sized sub-units would waste area.
3. **Chain the engines inside a stage:**
   `window unit → depthwise engine (bypassable) → dense engine → out`.
   A depthwise-separable block runs in one stage with both engines busy and no
   intermediate memory round-trip; a standard convolution bypasses the depthwise engine.
4. **Configurable engine-to-engine routing**, so patterns such as MobileNetV2's
   pointwise → depthwise → pointwise can be mapped.
5. **Compiler decisions per layer:** which engines are enabled, bypassed, or clock-gated,
   and in which mode (kernel size, groups, stride, fetch mode).

## Multi-model co-scheduling with provable non-interference

Idle engines are filled by other models: in the same stage, model A's depthwise layer
can run on the depthwise engine while model B's pointwise layer runs on the dense
engine. Interference does not disappear automatically, because engines still share
stage SRAM banks, input/output ports and routing, and weight-memory bandwidth. Because the
compiler knows the complete static schedule, it can **verify that no two models use the
same bank, port, or bandwidth slot in the same cycle**, which gives a proof of
non-interference and preserves per-model worst-case latency bounds (RQ1 of the proposal).

## Related work to position against

- Herald (Kwon et al., HPCA 2021): heterogeneous-dataflow sub-accelerators with
  multi-DNN layer scheduling. It works at chip level and schedules at runtime. This idea
  works per stage, inside a spatial pipeline, with static scheduling and latency bounds.
- Separate depthwise and pointwise engines in FPGA MobileNet accelerators
  (for example, Bai et al., IEEE TCAS-II 2018).

## Next step

Extend the Python model to count per-stage engine utilization for MobileNetV1 alone and for
MobileNetV1 + DS-CNN under three layouts: three equal sub-units, the chained
two-engine stage, and the chained stage with multi-model filling.

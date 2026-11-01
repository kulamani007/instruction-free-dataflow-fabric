# Design note: external memory tier, double buffering, and polyphase stride handling

**Author:** Kulamani Rout · **First recorded:** 30 September 2026 (written up 1 November 2026) · **Status:** decisions agreed in discussion; polyphase mode not yet in the model

## System view: external DRAM plus on-chip SRAM

The fabric is part of a *system*, not a chip alone: a large external memory (DRAM) next to
the stages' on-chip SRAM. This is standard and necessary, because weights do not fit
on-chip for most models (MobileNetV1 is about 4.2 MB at INT8; edge SRAM is hundreds of KB
to a few MB), so weights are streamed from DRAM.

**Determinism risk.** DRAM latency varies (refresh, row-buffer misses, bank conflicts).
Decisions:
- **Prefetch with double buffering**, so compute never waits on DRAM: while a stage computes
  from one buffer, the next weights or tiles load into the other.
- **Bound with worst-case DRAM timing**, not average timing.
- Study **predictable DRAM controllers** from the real-time systems literature (e.g. TDM or
  closed-page controllers) and give each model **fixed bandwidth slots** chosen by the
  compiler. These transfers are where both energy and determinism are won or lost.
- The ZU19EG board's DDR4 is a natural testbed for this tier.

**Correction recorded.** In the window-extraction model, bank activity depends only on the
kernel size relative to the bank grid, not on stride or image size; the banked fetch
already handles stride 2 with no conflicts.

## Strided windows: im2col-style duplication vs polyphase

**Idea considered:** extra memories holding pre-arranged stride windows (e.g. `[00,01,02]`,
`[02,03,04]`). This is essentially **im2col**, which duplicates data: **2.21×** storage for a
3×3 stride-2 layer at 112×112, **8.39×** at stride 1 (56×56), plus extra DRAM traffic.

**Better version: polyphase (space-to-depth).** Split the image into 4 phase sub-images
(even/odd rows × even/odd columns), storing each pixel **once** (1.00×). A stride-2 3×3
convolution becomes 4 stride-1 convolutions with sub-kernels 2×2, 2×1, 1×2 and 1×1, summed.
Every layer becomes stride 1, and the DMA can write the phase-split layout while copying in
from DRAM, as a per-layer compiler decision.

Caveat: the uneven sub-kernels complicate MAC utilisation; measure it before claiming it.

Bit-exact check (verified 30 September 2026):

```python
import numpy as np
def conv(x, w, s):
    Kr, Kc = w.shape; H = (x.shape[0]-Kr)//s + 1; W = (x.shape[1]-Kc)//s + 1
    return np.array([[(x[i*s:i*s+Kr, j*s:j*s+Kc]*w).sum() for j in range(W)] for i in range(H)])
x = np.random.randint(0, 9, (10, 10)); w = np.random.randint(-3, 4, (3, 3)); ref = conv(x, w, 2)
out = np.zeros_like(ref)
for p in (0, 1):
    for q in (0, 1):
        y = conv(x[p::2, q::2], w[p::2, q::2], 1); out += y[:ref.shape[0], :ref.shape[1]]
assert (out == ref).all()
```

## Next step

Add a polyphase mode to `model/window_bank_model.py` and compare SRAM and DRAM reads with
FULL and SHIFT on MobileNetV1's stride-2 layers.

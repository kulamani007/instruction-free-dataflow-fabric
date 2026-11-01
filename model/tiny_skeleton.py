"""
Tiny skeleton v0: three custom CNNs (5-8 layers each) resident at once  (v1)

Three small CNNs for three different edge tasks are mapped onto ONE skeleton
sized so that all of them are circuits at the same time: every layer of every
model has its own engine and its weights stay on-chip.  Nothing is shared
between the models except the skeleton itself, so no model can delay another.

  VisionNet  32x32x3 image classification (CIFAR-10 style)        6 layers
  KWSNet     49x10 MFCC keyword spotting (Speech Commands style)  6 layers
  PersonNet  96x96x1 person detection (visual wake words style)   8 layers

Engines are identical (fixed silicon must serve future models too): each has
M INT8 MACs, a weight SRAM, a line buffer and the 5x5 window unit.  The script
sizes the skeleton, checks it against the ZU19EG FPGA, and compares worst-case
latency with a time-shared NPU that has the same total MACs.

Simplifications: models are untrained (shapes only); frame latency is taken
as the sum of layer times (no overlap between layers inside one frame, so it is
an upper bound; line-buffer streaming would overlap them); global pooling is
folded into the FC engine; FPGA block RAM is counted in 4 KB data units.
"""
import math
import numpy as np
import composability_model as cm

F_CLK = 200e6                                  # FPGA prototype clock
ZU19EG = dict(dsp=1968, bram_kb=34.6e6 / 8 / 1024, uram_kb=36.0e6 / 8 / 1024)

# layer = (name, kind, k, stride, Hin, Win, Cin, Cout)
def net(spec):
    out = []
    for name, kind, k, s, h, w, ci, co in spec:
        ho, wo = math.ceil(h / s), math.ceil(w / s)
        if kind == "conv":   macs, wts = ho*wo*k*k*ci*co, k*k*ci*co
        elif kind == "dw":   macs, wts = ho*wo*k*k*ci, k*k*ci
        elif kind == "pw":   macs, wts = ho*wo*ci*co, ci*co
        else:                macs, wts = ci*co + h*w*ci, ci*co      # gap+fc
        line = k * w * ci if kind in ("conv", "dw") else (w * ci if kind == "pw" else ci)
        out.append(dict(name=name, kind=kind, k=k, macs=macs, w=wts, line=line))
    return out

MODELS = {
    "VisionNet": (net([("c1", "conv", 3, 1, 32, 32, 3, 16), ("c2", "conv", 3, 2, 32, 32, 16, 32),
                       ("c3", "conv", 3, 1, 16, 16, 32, 32), ("c4", "conv", 3, 2, 16, 16, 32, 64),
                       ("c5", "conv", 3, 1, 8, 8, 64, 64), ("fc", "fc", 1, 1, 8, 8, 64, 10)]), 1/30),
    "KWSNet":    (net([("c1", "conv", 3, 2, 49, 10, 1, 32), ("dw1", "dw", 3, 1, 25, 5, 32, 32),
                       ("pw1", "pw", 1, 1, 25, 5, 32, 64), ("dw2", "dw", 3, 1, 25, 5, 64, 64),
                       ("pw2", "pw", 1, 1, 25, 5, 64, 64), ("fc", "fc", 1, 1, 25, 5, 64, 12)]), 1/50),
    "PersonNet": (net([("c1", "conv", 3, 2, 96, 96, 1, 8), ("dw1", "dw", 3, 1, 48, 48, 8, 8),
                       ("pw1", "pw", 1, 1, 48, 48, 8, 16), ("dw2", "dw", 3, 2, 48, 48, 16, 16),
                       ("pw2", "pw", 1, 1, 24, 24, 16, 32), ("dw3", "dw", 3, 2, 24, 24, 32, 32),
                       ("pw3", "pw", 1, 1, 12, 12, 32, 64), ("fc", "fc", 1, 1, 12, 12, 64, 2)]), 1/15),
}

def pow2_kb(b): return 2 ** math.ceil(math.log2(max(b, 1024) / 1024))

if __name__ == "__main__":
    layers = [l for L, _ in MODELS.values() for l in L]
    E = len(layers)
    print("Workloads")
    for name, (L, p) in MODELS.items():
        print(f"  {name:<10}{len(L):>2} layers  {sum(l['macs'] for l in L)/1e6:6.2f} MMAC"
              f"  {sum(l['w'] for l in L)/1e3:6.1f} KB weights  every {p*1e3:4.0f} ms")
    w_max = max(l["w"] for l in layers); w_tot = sum(l["w"] for l in layers)
    lb_max = max(l["line"] for l in layers)
    w_kb, lb_kb = pow2_kb(w_max), pow2_kb(lb_max)
    print(f"\nSkeleton v0: {E} identical engines (one per layer)")
    print(f"  weight SRAM per engine {w_kb} KB (largest layer {w_max/1024:.1f} KB);"
          f" total {E*w_kb} KB, holding {w_tot/1024:.0f} KB -> {w_tot/(E*w_kb*1024):.0%} used")
    print(f"  line buffer per engine {lb_kb} KB (largest {lb_max/1024:.2f} KB)")
    print(f"  ops used: conv, depthwise, pointwise, global pool + FC, ReLU; kernels <= 3x3 (fits 5x5 window unit)")
    print("\n  Smaller engine SRAM, big layers spanning several engines (output channels split):")
    for kb in (4, 8, 16, 32, 64):
        n = sum(max(1, math.ceil(l["w"] / (kb * 1024))) for l in layers)
        print(f"    {kb:>2} KB/engine: {n:>3} engines, {n*kb:>5} KB weight SRAM, {w_tot/(n*kb*1024):4.0%} used")

    print(f"\nMACs per engine vs latency @ {F_CLK/1e6:.0f} MHz (frame latency = sum of layer times, upper bound)")
    print(f"  {'M':>4}{'DSPs':>7}{'ZU19EG':>8}  " + "".join(f"{n:>22}" for n in MODELS))
    for M in (4, 8, 16, 32, 64):
        cells = ""
        for name, (L, p) in MODELS.items():
            lat = sum(l["macs"] for l in L) / (M * F_CLK)
            fps = M * F_CLK / max(l["macs"] for l in L)
            cells += f"{lat*1e3:8.3f} ms {fps:>9.0f} fps"
        print(f"  {M:>4}{E*M:>7}{E*M/ZU19EG['dsp']:>7.0%}  {cells}")
    mem_kb = E * (w_kb + lb_kb)
    print(f"  on-chip memory {mem_kb} KB = {mem_kb/ZU19EG['bram_kb']:.0%} of ZU19EG block RAM"
          f" ({ZU19EG['bram_kb']:.0f} KB; UltraRAM {ZU19EG['uram_kb']:.0f} KB unused)")

    # ---- versus a time-shared NPU with the same total MACs
    M = 16
    cm.MAC_RATE = E * M * F_CLK
    models = [(n, [(l["name"], l["macs"], 0) for l in L], p) for n, (L, p) in MODELS.items()]
    print(f"\nWorst-case latency, M = {M} (skeleton {E*M} MACs vs time-shared NPU with the same {E*M} MACs)")
    print(f"  {'model':<11}{'skeleton (fixed)':>18}{'NPU alone':>12}{'NPU co-run max':>16}{'inflation':>11}")
    rng = np.random.default_rng(0)
    co = [[] for _ in models]
    for _ in range(40):
        ph = [rng.uniform(0, p) for _, _, p in models]
        for i, l in enumerate(cm.simulate(models, "B1-RM", [(1, 1)] * 3, ph, 1.0, 0.1)):
            co[i] += l
    for i, (name, (L, p)) in enumerate(MODELS.items()):
        sk = sum(l["macs"] for l in L) / (M * F_CLK)
        alone = sum(l["macs"] for l in L) / cm.MAC_RATE
        print(f"  {name:<11}{sk*1e3:>15.3f} ms{alone*1e3:>9.3f} ms{max(co[i])*1e3:>13.3f} ms"
              f"{max(co[i])/alone:>10.1f}x")

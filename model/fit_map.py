"""
Skeleton fit map: which CNNs can a fixed skeleton become, and at what cost?  (v1)

A skeleton is E identical engines; each engine has a MAC array and its own
weight SRAM.  A toy compiler splits a model's layer sequence into contiguous
groups and gives each group n >= 1 engines (output channels split across them),
minimising the slowest stage (= maximising pipeline throughput).

Fit classes (all three run with no processor and no instructions; only the
cost changes):
  CIRCUIT  every layer has its own engine(s) and its weights stay on-chip:
           the model literally becomes a circuit.
  FOLDED   weights still fit on-chip, but there are more layers than engines,
           so some engines serve several layers in a fixed (static) rotation.
  STREAMED weights do not fit on-chip and are re-read from DRAM every frame.
Separately, the model's operators are checked against the proposal's
section 3.1 op set and the window unit's largest kernel.

Layer tables come from model/workloads/*.csv (exact shapes, extracted from the
reference implementations by model/workloads/extract_layers.py).

Not modelled yet: activation memory (line buffers and, more importantly, the
buffers that hold skip connections until they are merged; long skips in YOLO
may need whole feature maps), per-operator array utilisation (depthwise on a
dense array is assumed 100% efficient), limits on how finely a layer can be
split across engines, and reconfiguration time in FOLDED mode.
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "workloads")
F_CLK = 500e6

SKELETONS = [   # name, engines, MACs per engine, weight SRAM per engine (bytes), DRAM B/s
    ("Small",   16, 128, 128 * 1024,  3.2e9),
    ("Medium",  64, 128, 128 * 1024,  6.4e9),
    ("Large",  128, 256, 256 * 1024, 12.8e9),
]
K_MAX = 5                    # window unit bank grid 5x5 (proposal section 3.2)
SUPPORTED = {"residual add", "global avgpool", "adaptive avgpool", "avgpool",
             "maxpool 2x2", "maxpool 3x3", "maxpool 5x5"}         # plus conv, dw, matmul, activation
ORDER = ["DS-CNN", "SqueezeNet1.1", "MobileNetV3-Small", "MobileNetV2", "YOLOv8n",
         "MobileNetV1", "EfficientNet-B0", "ResNet-18", "YOLOv8s", "ResNet-50", "VGG-16"]

def load(name):
    with open(os.path.join(WL, f"{name}.csv")) as f:
        rows = [{k: (v if k == "kind" else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    return [(r["macs"], r["params"]) for r in rows], rows        # INT8: params == bytes

def min_engines(layers, T, rate, sram, single):
    """Fewest engines covering all layers if no stage may take longer than T (contiguous DP)."""
    n, INF = len(layers), float("inf")
    dp = [0] + [INF] * n
    for i in range(1, n + 1):
        macs = w = 0
        for j in range(i - 1, -1, -1):                 # group = layers j .. i-1
            macs += layers[j][0]; w += layers[j][1]
            need = max(1, math.ceil(macs / (rate * T) - 1e-9),
                       math.ceil(w / sram) if sram else 1)
            dp[i] = min(dp[i], dp[j] + need)
            if single: break
    return dp[n]

def best_period(layers, E, rate, sram, single):
    """Smallest stage time T (s) achievable with E engines, or None if it cannot fit."""
    total = sum(m for m, _ in layers)
    lo, hi = total / (E * rate), total / rate * 1.0001
    if min_engines(layers, hi * 64, rate, sram, single) > E:      # infeasible even when slow
        return None
    hi *= 64
    for _ in range(60):
        mid = (lo + hi) / 2
        if min_engines(layers, mid, rate, sram, single) <= E: hi = mid
        else: lo = mid
    return hi

def fit(name, skel, info):
    _, E, mpe, sram, bw = skel
    layers, rows = load(name)
    rate = mpe * F_CLK
    total_macs = sum(m for m, _ in layers)
    weights = sum(w for _, w in layers)
    io = rows[0]["hin"] * rows[0]["win"] * rows[0]["cin"] + rows[-1]["cout"]
    T = best_period(layers, E, rate, sram, single=True)
    cls, dram = "CIRCUIT", io
    Tf = best_period(layers, E, rate, sram, single=False)    # same model, folding allowed
    fold_gain = T / Tf if T and Tf else None
    if T is None:
        T, cls = best_period(layers, E, rate, sram, single=False), "FOLDED"
    if T is None:
        Tc = best_period(layers, E, rate, None, single=False)
        T, cls, dram = max(Tc, (weights + io) / bw), "STREAMED", weights + io
    ext = sorted(o for o in info["ops"] if not o.startswith("act:") and o not in SUPPORTED)
    big = sorted({f"{r['kh']}x{r['kw']}" for r in rows if max(r["kh"], r["kw"]) > K_MAX})
    if big: ext.append("kernel " + "/".join(big) + f" > {K_MAX}x{K_MAX}")
    return dict(cls=cls, fps=1 / T, util=total_macs / (E * rate * T), dram=dram,
                weights=weights, layers=len(layers), ext=ext, fold_gain=fold_gain)

def fit_all():
    info = json.load(open(os.path.join(WL, "ops.json")))
    return info, {(m, s[0]): fit(m, s, info[m]) for m in ORDER for s in SKELETONS}

if __name__ == "__main__":
    info, res = fit_all()
    for name, E, mpe, sram, bw in SKELETONS:
        print(f"{name:<7}: {E:>3} engines x {mpe} MACs = {E*mpe:>5} MACs "
              f"({2*E*mpe*F_CLK/1e12:4.1f} TOPS INT8), weight SRAM {E*sram/2**20:4.0f} MB, "
              f"DRAM {bw/1e9:4.1f} GB/s")
    print(f"\n{'model':<18}{'layers':>7}{'weights':>9}{'GMAC':>7}  " +
          "".join(f"{s[0]:^30}" for s in SKELETONS) + "  ops beyond proposal op set")
    for m in ORDER:
        r0 = res[(m, SKELETONS[0][0])]
        cells = "".join(f"{res[(m, s[0])]['cls']:<9}{res[(m, s[0])]['fps']:>8.0f} fps"
                        f"{res[(m, s[0])]['util']:>6.0%} {res[(m, s[0])]['dram']/1e6:>4.1f}MB "
                        for s in SKELETONS)
        print(f"{m:<18}{r0['layers']:>7}{r0['weights']/1e6:>8.2f}M{info[m]['macs']/1e9:>7.2f}  "
              f"{cells} {', '.join(r0['ext']) or '-'}")
    print("\ncell = fit class, frames/s (upper bound), MAC utilisation, DRAM traffic per frame")
    print("\nCIRCUIT models: throughput gain if the compiler may fold instead (same skeleton)")
    for (m, s), r in res.items():
        if r["cls"] == "CIRCUIT":
            print(f"  {m:<18}{s:<8} circuit {r['util']:4.0%} MAC utilisation, folding x{r['fold_gain']:.2f}")

"""
Composability experiment: does co-running models change a model's latency?  (v1)

Three heterogeneous edge models share one accelerator with fixed total
resources (MAC array, DRAM bandwidth).  Each model runs periodically.  We
measure every model's inference latency running ALONE and CO-RUNNING under:

  B1  time-shared   : the whole array runs one layer at a time (non-preemptive,
                      layer-granular), picked FIFO or rate-monotonic priority.
                      The usual single-NPU runtime.
  B2  split-compute : each model owns a fixed slice of the MAC array, but DRAM
                      bandwidth is shared, work-conserving (max-min fair).
                      Spatial multi-tenancy without bandwidth isolation.
  P   proposed      : the same compute slices as B2, plus a fixed TDM share of
                      DRAM bandwidth per model (non-work-conserving).  Shares
                      are chosen at compile time by a small search (toy compiler).

Layer cost (identical for all systems): a layer needs MACs and DRAM bytes (INT8
weights + input + output activations, no fusion).  With double buffering it
progresses at min(compute rate, bandwidth rate).  Fluid, event-driven model.

Read the results with these limits in mind:
  - P's isolation holds here by construction (its rates depend only on its own
    shares).  What this model measures is HOW LARGE the interference is in the
    baselines on real layer shapes, and WHAT ISOLATION COSTS.  Proving that
    hardware keeps the isolation is the RTL/FPGA step.
  - Not modelled yet: DRAM timing variation (refresh, row misses), per-operator
    array utilisation (all ops at 100%), on-chip activation reuse, dispatch
    overhead (0 for the baselines, so they are not penalised).
"""
import numpy as np, itertools, math
from collections import deque

# ---------------------------------------------------------------- hardware ----
F_CLK  = 500e6                 # Hz
MACS   = 256                   # INT8 MACs/cycle in total (16x16 array)
MAC_RATE = MACS * F_CLK        # MAC/s
DRAM_BW  = 3.2e9               # B/s, LPDDR4 x16 class
G      = 32                    # compiler allocation granularity (1/32 of each)
TDM_FRAME = 64 / F_CLK         # TDM frame of 64 cycles

# --------------------------------------------------------------- workloads ----
# layer = (name, MACs, DRAM bytes)    INT8: bytes = params + in act + out act
def conv(n, Hi, Wi, Ci, Co, kh, kw, s):
    Ho, Wo = math.ceil(Hi / s), math.ceil(Wi / s)
    return (n, Ho*Wo*kh*kw*Ci*Co, kh*kw*Ci*Co + Hi*Wi*Ci + Ho*Wo*Co)

def dw(n, Hi, Wi, C, k, s):
    Ho, Wo = math.ceil(Hi / s), math.ceil(Wi / s)
    return (n, Ho*Wo*k*k*C, k*k*C + Hi*Wi*C + Ho*Wo*C)

def mm(n, T, Di, Do, weights=True):          # token-wise linear layer
    return (n, T*Di*Do, (Di*Do if weights else 0) + T*Di + T*Do)

def mobilenet_v1():                          # 224x224, alpha 1 (detection backbone proxy)
    L, H = [conv("conv1", 224, 224, 3, 32, 3, 3, 2)], 112
    cfg = [(32,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,512,2)] + \
          [(512,512,1)]*5 + [(512,1024,2),(1024,1024,1)]
    for i, (ci, co, s) in enumerate(cfg, 1):
        L.append(dw(f"dw{i}", H, H, ci, 3, s)); H = math.ceil(H / s)
        L.append(conv(f"pw{i}", H, H, ci, co, 1, 1, 1))
    L.append(mm("fc", 1, 1024, 1000))
    return L

def ds_cnn():                                # MLPerf Tiny KWS, 49x10 MFCC
    L = [conv("conv1", 49, 10, 1, 64, 10, 4, 2)]
    for i in range(1, 5):
        L += [dw(f"dw{i}", 25, 5, 64, 3, 1), conv(f"pw{i}", 25, 5, 64, 64, 1, 1, 1)]
    L.append(mm("fc", 1, 64, 12))
    return L

def kwt1():                                  # KWT-1: d=64, 1 head, MLP 256, 12 blocks
    T, d = 99, 64
    L = [mm("embed", 98, 40, d)]
    for b in range(12):
        L += [mm(f"b{b}.qkv", T, d, 3*d),
              (f"b{b}.qk", T*T*d, 2*T*d + T*T),          # scores: no weights
              (f"b{b}.av", T*T*d, T*T + 2*T*d),
              mm(f"b{b}.proj", T, d, d),
              mm(f"b{b}.mlp1", T, d, 4*d), mm(f"b{b}.mlp2", T, 4*d, d)]
    L.append(mm("head", 1, d, 12))
    return L

MODELS = [("MobileNetV1", mobilenet_v1(), 1/30),     # vision, 30 FPS
          ("DS-CNN",      ds_cnn(),       1/25),     # keyword spotting, 40 ms hop
          ("KWT-1",       kwt1(),         1/10)]     # audio transformer, 10 Hz

# ------------------------------------------------------------ toy compiler ----
def layer_time(layer, mac_rate, bw):
    _, macs, byts = layer
    return max(macs / mac_rate, byts / bw if byts else 0.0)

def p_latency(layers, m, b):
    """P's latency for given shares (m/G of MACs, b/G of bandwidth), incl. worst TDM wait."""
    return sum(layer_time(l, MAC_RATE*m/G, DRAM_BW*b/G) for l in layers) \
           + len(layers) * TDM_FRAME * (1 - b/G)

def full_chip(layers):
    return sum(layer_time(l, MAC_RATE, DRAM_BW) for l in layers)

def compile_shares(models, objective="deadline"):
    """Pick (m_i, b_i) for 3 models by exhaustive search.
    deadline : minimise max_i latency_i / period_i        (only cares about deadlines)
    fair     : minimise max_i latency_i / full_chip_i, subject to latency_i <= period_i
               (spreads the slowdown evenly across models)"""
    tab = [{(m, b): p_latency(L, m, b) for m in range(1, G) for b in range(1, G)}
           for _, L, _ in models]
    norm = [p if objective == "deadline" else full_chip(L) for _, L, p in models]
    best = None
    for m1, m2 in itertools.product(range(1, G), repeat=2):
        m3 = G - m1 - m2
        if m3 < 1: continue
        for b1, b2 in itertools.product(range(1, G), repeat=2):
            b3 = G - b1 - b2
            if b3 < 1: continue
            sh = [(m1, b1), (m2, b2), (m3, b3)]
            if any(tab[i][sh[i]] > models[i][2] for i in range(3)): continue
            score = max(tab[i][sh[i]] / norm[i] for i in range(3))
            if best is None or score < best[0]:
                best = (score, sh)
    return best[1], best[0]

# --------------------------------------------------------------- simulator ----
def maxmin(demand, cap):
    alloc, left = {}, cap
    items = sorted(demand.items(), key=lambda kv: kv[1])
    for k, (j, d) in enumerate(items):
        alloc[j] = min(d, left / (len(items) - k)); left -= alloc[j]
    return alloc

def simulate(models, system, shares, phases, horizon, warmup):
    """Event-driven run.  Returns per-model list of latencies (s)."""
    n = len(models)
    period = [p for _, _, p in models]
    layers = [L for _, L, _ in models]
    nxt = list(phases)
    queue = [deque() for _ in range(n)]
    cur, rem, start, wait = [None]*n, [0.0]*n, [0.0]*n, [0.0]*n
    owner = None                                     # B1: model holding the array
    lat = [[] for _ in range(n)]
    slot_off = np.cumsum([0] + [b for _, b in shares])[:-1] / G * TDM_FRAME
    t, eps = 0.0, 1e-15

    def begin_layer(j):
        if system == "P":                            # wait for own TDM slot
            frac = shares[j][1] / G
            ph = (t - slot_off[j]) % TDM_FRAME
            wait[j] = t + (0.0 if ph < frac*TDM_FRAME else TDM_FRAME - ph)
        rem[j] = 1.0

    while True:
        for j in range(n):                           # arrivals
            while nxt[j] <= t + eps and nxt[j] < horizon:
                queue[j].append(nxt[j]); nxt[j] += period[j]
            if cur[j] is None and queue[j]:
                start[j], cur[j] = queue[j].popleft(), 0; begin_layer(j)
        if system.startswith("B1") and owner is None:
            ready = [j for j in range(n) if cur[j] is not None]
            if ready:
                key = (lambda j: start[j]) if system == "B1-FIFO" else (lambda j: period[j])
                owner = min(ready, key=key)

        rate = {}                                    # fraction of layer per second
        busy = [j for j in range(n) if cur[j] is not None and wait[j] <= t + eps]
        if system.startswith("B1"):
            if owner in busy:
                rate[owner] = 1 / layer_time(layers[owner][cur[owner]], MAC_RATE, DRAM_BW)
        elif system == "B2":
            mr = {j: MAC_RATE * shares[j][0] / G for j in busy}
            need = {j: layers[j][cur[j]][2] * mr[j] / layers[j][cur[j]][1] for j in busy}
            bw = maxmin(need, DRAM_BW)
            for j in busy:
                _, macs, byts = layers[j][cur[j]]
                rate[j] = min(mr[j] / macs, bw[j] / byts if byts else math.inf)
        else:                                        # P: own shares only
            for j in busy:
                rate[j] = 1 / layer_time(layers[j][cur[j]], MAC_RATE*shares[j][0]/G,
                                         DRAM_BW*shares[j][1]/G)

        cand = [x - t for x in nxt if x < horizon] + \
               [wait[j] - t for j in range(n) if cur[j] is not None and wait[j] > t + eps] + \
               [rem[j] / r for j, r in rate.items()]
        if not cand: break
        dt = max(min(cand), 0.0)
        t += dt
        for j, r in rate.items():
            rem[j] -= r * dt
            if rem[j] <= 1e-9:                       # layer done
                if system.startswith("B1"): owner = None
                cur[j] += 1
                if cur[j] == len(layers[j]):
                    if start[j] >= warmup: lat[j].append(t - start[j])
                    cur[j] = None
                    if queue[j]: start[j], cur[j] = queue[j].popleft(), 0; begin_layer(j)
                else:
                    begin_layer(j)
    return lat

def run(models, system, shares, trials, horizon=1.0, warmup=0.2, alone=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = range(len(models)) if alone is None else [alone]
    sub = [models[i] for i in idx]
    sh = [shares[i] for i in idx]
    out = [[] for _ in idx]
    for _ in range(trials):
        ph = [rng.uniform(0, p) for _, _, p in sub]
        for k, l in enumerate(simulate(sub, system, sh, ph, horizon, warmup)):
            out[k] += l
    return [np.array(x) for x in out]

# -------------------------------------------------------------- experiment ----
SCENARIOS = [
    ("A: vision + KWS + audio transformer", MODELS),
    ("B: two cameras + KWS (DRAM-heavy)",
     [("MobileNetV1-a", mobilenet_v1(), 1/30), ("MobileNetV1-b", mobilenet_v1(), 1/30),
      ("DS-CNN", ds_cnn(), 1/25)]),
]
SYSTEMS = ["B1-FIFO", "B1-RM", "B2", "P-deadline", "P-fair"]

def evaluate(models, trials=40):
    """{(model, system): dict(alone, mean, p99, max, miss, bound)}.  B2 uses P-fair's compute split."""
    shares = {"deadline": compile_shares(models, "deadline")[0],
              "fair": compile_shares(models, "fair")[0]}
    res = {}
    for label in SYSTEMS:
        system = "P" if label.startswith("P") else label
        sh = shares["deadline" if label == "P-deadline" else "fair"]
        co = run(models, system, sh, trials)
        for i, (name, L, period) in enumerate(models):
            al = run(models, system, sh, 5, alone=i)[0]
            res[(name, label)] = dict(alone=al.max(), mean=co[i].mean(),
                                      p99=np.percentile(co[i], 99), max=co[i].max(),
                                      miss=np.mean(co[i] > period),
                                      bound=p_latency(L, *sh[i]) if system == "P" else None)
    return shares, res

if __name__ == "__main__":
    print(f"Hardware: {MACS} MACs @ {F_CLK/1e6:.0f} MHz, DRAM {DRAM_BW/1e9:.1f} GB/s")
    for name, L, p in MODELS:
        print(f"  {name:<12} {len(L):>3} layers  {sum(l[1] for l in L)/1e6:7.1f} MMAC"
              f"  {sum(l[2] for l in L)/1e6:6.2f} MB DRAM  period {p*1e3:5.1f} ms"
              f"  alone on full chip {full_chip(L)*1e3:6.3f} ms")

    for title, models in SCENARIOS:
        shares, res = evaluate(models)
        print(f"\n=== Scenario {title} ===")
        for obj, sh in shares.items():
            print(f"  compiler '{obj}' shares (MAC/{G}, BW/{G}): "
                  + ", ".join(f"{m[0]} {s}" for m, s in zip(models, sh)))
        print(f"  {'model':<14}{'system':<11}{'alone':>10}{'co-run mean':>13}{'p99':>10}"
              f"{'max':>10}{'max/alone':>10}{'bound':>10}{'misses':>8}")
        for name, _, _ in models:
            for label in SYSTEMS:
                r = res[(name, label)]
                bound = f"{r['bound']*1e3:8.3f}ms" if r["bound"] is not None else f"{'-':>10}"
                print(f"  {name:<14}{label:<11}{r['alone']*1e3:8.3f}ms{r['mean']*1e3:11.3f}ms"
                      f"{r['p99']*1e3:8.3f}ms{r['max']*1e3:8.3f}ms{r['max']/r['alone']:9.2f}x"
                      f"{bound}{r['miss']:>8.1%}")
            print()

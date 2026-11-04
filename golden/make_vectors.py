"""
Generate parameters, configuration and test vectors for the RTL (hardware step H0)  (v1)

For each tiny-skeleton model this script
  1. draws deterministic int8 weights (seeded; untrained: bit-exactness does not need training),
  2. calibrates every requantisation (M, S) and bias on a separate calibration input, so
     activations use the int8 range instead of collapsing to 0 or saturating,
  3. runs the golden model on N_TEST test inputs,
  4. writes golden/vectors/<model>/: config.json plus one $readmemh-ready hex file per tensor:
       input_<n>.hex                      int8 input, H x W x C order
       <layer>.weights.hex / .bias.hex    int8 weights (layout in golden_model.py) / int32 bias
       <layer>.M.hex / .S.hex             per-channel requantisation multiplier / shift
       <layer>.acc_<n>.hex                int32 accumulator before requantisation
       <layer>.out_<n>.hex                int8 layer output (for "fc": the logits)
       <layer>.pool_acc_<n>.hex, .pool_<n>.hex, .pool_M.hex, .pool_S.hex   ("fc" only)
     Hex is two's complement, fixed width (int8: 2 digits, int32: 8, M: 4, S: 2), one per line.
  5. compares a SHA-256 of every file with golden/vectors.sha256 (committed), so any change
     to the arithmetic or the generator is caught.  --update-manifest rewrites it.

  python golden/make_vectors.py [--update-manifest]
"""
import hashlib, json, math, os, sys
import numpy as np
from golden_model import MODELS, CONTRACT_VERSION, forward, input_shape, layer_forward, out_size

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vectors")
MANIFEST = os.path.join(HERE, "vectors.sha256")
N_TEST = 2
SEED = {"VisionNet": 101, "KWSNet": 202, "PersonNet": 303}
TARGET = 100                     # calibrated 99.9th percentile maps to this int8 value

def encode(m):
    """Real multiplier m > 0  ->  (M, S) with m ~= M / 2^S, M in [16384, 32767]."""
    S = 14 - math.floor(math.log2(m))
    M = round(m * 2 ** S)
    if M == 32768: M, S = 16384, S - 1
    assert 16384 <= M <= 32767 and 1 <= S <= 47, (m, M, S)
    return M, S

def calibrate(acc, relu):
    """Per-channel (M, S) arrays mapping the channel's 99.9th percentile to TARGET."""
    a = acc.reshape(-1, acc.shape[-1]).astype(np.float64)
    ref = np.percentile(np.maximum(a, 0) if relu else np.abs(a), 99.9, axis=0)
    ref = np.where(ref > 0, ref, np.maximum(np.abs(a).max(axis=0), 1))
    MS = [encode(TARGET / r) for r in ref]
    return np.array([m for m, _ in MS]), np.array([s for _, s in MS])

def rand_input(rng, model):
    return np.clip(np.round(rng.normal(0, 40, input_shape(model))), -128, 127).astype(np.int8)

def wshape(L):
    k, ci, co = L["k"], L["ci"], L["co"]
    return {"conv": (co, k, k, ci), "dw": (ci, k, k), "pw": (co, ci), "fc": (co, ci)}[L["kind"]]

N_CALIB = 16                     # calibration inputs (separate from the test inputs)

def build(model):
    """Deterministic parameters for a model, calibrated layer by layer on N_CALIB inputs."""
    rng = np.random.default_rng(SEED[model])
    xs = [rand_input(rng, model) for _ in range(N_CALIB)]
    params = []
    for L in MODELS[model][0]:
        P = {"w": np.clip(np.round(rng.normal(0, 40, wshape(L))), -127, 127).astype(np.int8)}
        co = L["co"]
        P["b"] = np.zeros(co, np.int64)
        if L["kind"] == "fc":
            pool = np.stack([x.astype(np.int64).sum(axis=(0, 1)) for x in xs])
            P["pool_M"], P["pool_S"] = calibrate(pool, relu=True)
        P["M"], P["S"] = np.full(co, 16384), np.full(co, 47)         # placeholders
        acc = np.concatenate([layer_forward(L, P, x)[0].reshape(-1, co) for x in xs])
        P["b"] = np.round(rng.normal(0, 0.25 * (acc.std(axis=0) + 1))).astype(np.int64)
        P["M"], P["S"] = calibrate(acc + P["b"], relu=L["kind"] != "fc")
        xs = [layer_forward(L, P, x)[1] for x in xs]
        params.append(P)
    tests = [rand_input(rng, model) for _ in range(N_TEST)]
    return params, tests

def worst_bits(L, P):
    """Accumulator width that can never overflow for ANY int8 input (signed bits)."""
    terms = {"conv": L["k"] * L["k"] * L["ci"], "dw": L["k"] * L["k"], "pw": L["ci"], "fc": L["ci"]}[L["kind"]]
    bound = terms * 127 * 128 + int(np.abs(P["b"]).max())
    return bound.bit_length() + 1

def hexfile(path, arr, width):
    mask = (1 << (4 * width)) - 1
    with open(path, "w") as f:
        f.writelines(f"{int(v) & mask:0{width}x}\n" for v in np.asarray(arr).ravel())

def write_model(model):
    d = os.path.join(OUT, model)
    os.makedirs(d, exist_ok=True)
    params, tests = build(model)
    traces = [forward(model, params, t) for t in tests]
    for n, t in enumerate(tests):
        hexfile(os.path.join(d, f"input_{n}.hex"), t, 2)
    cfg = dict(model=model, contract_version=CONTRACT_VERSION, input_shape_hwc=input_shape(model),
               n_tests=N_TEST, layers=[])
    print(f"\n{model}")
    for i, (L, P) in enumerate(zip(MODELS[model][0], params)):
        nm = L["name"]
        hexfile(os.path.join(d, f"{nm}.weights.hex"), P["w"], 2)
        hexfile(os.path.join(d, f"{nm}.bias.hex"), P["b"], 8)
        hexfile(os.path.join(d, f"{nm}.M.hex"), P["M"], 4)
        hexfile(os.path.join(d, f"{nm}.S.hex"), P["S"], 2)
        if L["kind"] == "fc":
            hexfile(os.path.join(d, f"{nm}.pool_M.hex"), P["pool_M"], 4)
            hexfile(os.path.join(d, f"{nm}.pool_S.hex"), P["pool_S"], 2)
        for n, tr in enumerate(traces):
            hexfile(os.path.join(d, f"{nm}.acc_{n}.hex"), tr[i]["acc"], 8)
            hexfile(os.path.join(d, f"{nm}.out_{n}.hex"), tr[i]["out"], 2)
            if tr[i]["pool"] is not None:
                hexfile(os.path.join(d, f"{nm}.pool_acc_{n}.hex"), tr[i]["pool"][0], 8)
                hexfile(os.path.join(d, f"{nm}.pool_{n}.hex"), tr[i]["pool"][1], 2)
        acc = np.concatenate([tr[i]["acc"].ravel() for tr in traces])
        out = np.concatenate([tr[i]["out"].ravel() for tr in traces])
        bits = int(np.abs(acc).max()).bit_length() + 1                # signed width needed
        relu = L["kind"] != "fc"
        ho = 1 if L["kind"] == "fc" else out_size(L["h"], L["k"], L["s"])
        wo = 1 if L["kind"] == "fc" else out_size(L["wd"], L["k"], L["s"])
        cfg["layers"].append(dict(name=nm, kind=L["kind"], k=L["k"], stride=L["s"], pad=L["k"] // 2,
                                  in_hwc=[L["h"], L["wd"], L["ci"]], out_hwc=[ho, wo, L["co"]],
                                  relu=relu, acc_bits_seen=bits, acc_bits_worst_case=worst_bits(L, P)))
        print(f"  {nm:<4} {L['kind']:<4} in {L['h']:>2}x{L['wd']:<2}x{L['ci']:<3} out {ho:>2}x{wo:<2}x{L['co']:<3}"
              f" acc bits seen {bits:>2} / worst case {worst_bits(L, P):>2}   zeros {np.mean(out == 0):5.1%}"
              f"   at max {np.mean(out == 127):4.1%}" + ("" if relu else f"   at min {np.mean(out == -128):4.1%}"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

def manifest():
    lines = []
    for root, _, files in sorted(os.walk(OUT)):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
    return "".join(lines)

if __name__ == "__main__":
    for m in MODELS:
        write_model(m)
    now = manifest()
    n = now.count("\n")
    if "--update-manifest" in sys.argv:
        open(MANIFEST, "w").write(now)
        print(f"\nmanifest written: {n} files")
    elif os.path.exists(MANIFEST) and open(MANIFEST).read() == now:
        print(f"\nvectors match manifest: {n} files identical")
    else:
        print("\nVECTORS DIFFER FROM MANIFEST (run with --update-manifest only if the change is intended)")
        sys.exit(1)

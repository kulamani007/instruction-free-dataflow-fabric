"""
Self-tests for the golden model: each check compares it with an independent reference.

  python golden/test_golden.py        (run make_vectors.py first; exits non-zero on failure)

  1. conv / dw / pw against plain nested loops (k = 1, 3, 5; stride 1, 2; odd sizes; extremes)
  2. requantisation against exact Python integer arithmetic (incl. negative and .5 cases)
  3. a conv layer computed by fetching every window through the banked window-extraction
     model (model/window_bank_model.py), i.e. the way the hardware will, equals the golden layer
  4. the hex files read back (sign-extended) equal a fresh golden run, with the right lengths
  5. MAC counts agree with model/tiny_skeleton.py
"""
import json, os, sys
import numpy as np
from golden_model import (MODELS, conv_acc, dw_acc, pw_acc, requant, forward, out_size)
from make_vectors import build, OUT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import window_bank_model as wb                                    # noqa: E402

rng = np.random.default_rng(7)
passed = 0

def ok(name):
    global passed
    passed += 1
    print(f"ok    {name}")

def naive(x, w, b, k, s, kind):
    H, W, C = x.shape
    p = k // 2
    ho, wo = out_size(H, k, s), out_size(W, k, s)
    co = w.shape[0]
    y = np.zeros((ho, wo, co), np.int64)
    for oh in range(ho):
        for ow in range(wo):
            for o in range(co):
                t = int(b[o])
                for i in range(k):
                    for j in range(k):
                        r, c = oh * s + i - p, ow * s + j - p
                        if 0 <= r < H and 0 <= c < W:
                            if kind == "conv":
                                t += sum(int(x[r, c, q]) * int(w[o, i, j, q]) for q in range(C))
                            else:
                                t += int(x[r, c, o]) * int(w[o, i, j])
                y[oh, ow, o] = t
    return y

def i8(shape, extreme=False):
    if extreme:
        return rng.choice(np.array([-128, 127], np.int8), size=shape)
    return rng.integers(-128, 128, shape, dtype=np.int8)

# 1 ------------------------------------------------------------------------------------
for (H, W, C, co, k, s, ext) in [(7, 5, 3, 4, 3, 1, False), (8, 9, 2, 3, 3, 2, False),
                                 (6, 6, 2, 2, 5, 1, False), (9, 7, 3, 2, 5, 2, False),
                                 (5, 4, 4, 3, 1, 1, False), (7, 7, 3, 2, 3, 2, True)]:
    x, b = i8((H, W, C), ext), rng.integers(-5000, 5000, co)
    w = rng.integers(-127, 128, (co, k, k, C), dtype=np.int8)
    assert np.array_equal(conv_acc(x, w, b, k, s), naive(x, w, b, k, s, "conv"))
    wd, bd = rng.integers(-127, 128, (C, k, k), dtype=np.int8), rng.integers(-5000, 5000, C)
    assert np.array_equal(dw_acc(x, wd, bd, k, s), naive(x, wd, bd, k, s, "dw"))
    if k == 1:
        wp = w[:, 0, 0, :]
        assert np.array_equal(pw_acc(x, wp, b), naive(x, w, b, 1, 1, "conv"))
ok("conv / dw / pw equal nested-loop reference (k 1/3/5, stride 1/2, odd sizes, extremes)")

# 2 ------------------------------------------------------------------------------------
acc = np.concatenate([rng.integers(-(1 << 31), 1 << 31, 20000), [0, 1, -1, 3, -3, 5, -5]])
M = rng.integers(16384, 32768, acc.size)
S = rng.integers(1, 48, acc.size)
for relu in (True, False):
    got = np.array([requant(np.array([a]), [m], [s], relu)[0] for a, m, s in zip(acc, M, S)])
    ref = [max(0 if relu else -128, min(127, (int(a) * int(m) + (1 << (int(s) - 1))) // (1 << int(s))))
           for a, m, s in zip(acc, M, S)]
    assert np.array_equal(got, ref)
# exact halves round up (towards +inf): -2.5 -> -2, 2.5 -> 3, -0.5 -> 0
assert [int(requant(np.array([v]), [16384], [15], False)[0]) for v in (-5, 5, -1)] == [-2, 3, 0]
ok("requantisation equals exact integer arithmetic (20,007 cases; halves round up)")

# 3 ------------------------------------------------------------------------------------
def banked_conv(x, w, b, k, s):
    """Conv where every window is fetched through the banked memory model, one channel at a time."""
    p = k // 2
    xp = np.pad(x.astype(np.int64), ((p, p), (p, p), (0, 0)))
    Hp, Wp, C = xp.shape
    banks = [wb.build_banks(xp[:, :, c], k, k) for c in range(C)]
    ho, wo = out_size(x.shape[0], k, s), out_size(x.shape[1], k, s)
    y = np.zeros((ho, wo, w.shape[0]), np.int64)
    for r0, c0 in wb.addr_gen(Hp, Wp, k, k, s):
        win = np.zeros((k, k, C), np.int64)
        for c in range(C):
            bk, LW = banks[c]
            used = set()
            for i in range(k):
                for j in range(k):
                    bank, v = wb.fetch_pixel(bk, LW, k, k, r0, c0, r0 + i, c0 + j)
                    assert bank not in used; used.add(bank)
                    win[i, j, c] = v
        y[r0 // s, c0 // s] = np.einsum("ijc,oijc->o", win, w.astype(np.int64)) + b
    return y

for model, li in (("KWSNet", 0), ("VisionNet", 0), ("PersonNet", 0)):
    params, tests = build(model)
    L, P = MODELS[model][0][li], params[li]
    acc_banked = banked_conv(tests[0], P["w"], P["b"], L["k"], L["s"])
    acc_golden = forward(model, params, tests[0])[li]["acc"]
    assert np.array_equal(acc_banked, acc_golden), model
ok("first conv layer of all 3 models via banked window fetch equals golden (stride 1 and 2)")

# 4 ------------------------------------------------------------------------------------
def readhex(path, width):
    v = np.array([int(l, 16) for l in open(path)], np.int64)
    sign = 1 << (4 * width - 1)
    return np.where(v >= sign, v - (1 << (4 * width)), v)

nfiles = 0
for model in MODELS:
    d = os.path.join(OUT, model)
    cfg = json.load(open(os.path.join(d, "config.json")))
    params, tests = build(model)
    for n, t in enumerate(tests):
        assert np.array_equal(readhex(os.path.join(d, f"input_{n}.hex"), 2), t.ravel()); nfiles += 1
        tr = forward(model, params, t)
        for i, L in enumerate(cfg["layers"]):
            out = readhex(os.path.join(d, f"{L['name']}.out_{n}.hex"), 2)
            acc = readhex(os.path.join(d, f"{L['name']}.acc_{n}.hex"), 8)
            assert out.size == int(np.prod(L["out_hwc"])), (model, L["name"])
            assert np.array_equal(out, tr[i]["out"].ravel()) and np.array_equal(acc, tr[i]["acc"].ravel())
            nfiles += 2
    for i, L in enumerate(cfg["layers"]):
        P = params[i]
        for key, wdt in (("weights", 2), ("bias", 8), ("M", 4), ("S", 2)):
            src = {"weights": P["w"], "bias": P["b"], "M": P["M"], "S": P["S"]}[key]
            assert np.array_equal(readhex(os.path.join(d, f"{L['name']}.{key}.hex"), wdt), np.ravel(src))
            nfiles += 1
ok(f"hex files read back equal a fresh golden run ({nfiles} files checked)")

# 5 ------------------------------------------------------------------------------------
for model in MODELS:
    for L in MODELS[model][0]:
        ho, wo = out_size(L["h"], L["k"], L["s"]), out_size(L["wd"], L["k"], L["s"])
        macs = {"conv": ho * wo * L["k"] ** 2 * L["ci"] * L["co"], "dw": ho * wo * L["k"] ** 2 * L["ci"],
                "pw": ho * wo * L["ci"] * L["co"], "fc": L["ci"] * L["co"] + L["h"] * L["wd"] * L["ci"]}[L["kind"]]
        assert macs == L["macs"], (model, L["name"], macs, L["macs"])
ok("golden layer shapes give the MAC counts used in model/tiny_skeleton.py")

print(f"ALL {passed} GOLDEN TESTS PASSED")

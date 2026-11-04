"""
Bit-exact INT8 golden model of the tiny-skeleton CNNs (hardware step H0)  (v1)

This is the reference the RTL must match bit for bit.  Network shapes come from
model/tiny_skeleton.py (one source of truth).

Arithmetic contract (version 1)
  activations  int8, zero point 0 (symmetric); tensor layout H x W x C, channel fastest
  weights      int8, symmetric, per output channel
                 conv [Cout][kh][kw][Cin]   dw [C][kh][kw]   pw / fc [Cout][Cin]
  bias         int32 per output channel, added to the accumulator
  accumulate   int32 (the model asserts it never overflows)
  requantize   y = (acc * M + 2^(S-1)) >> S, per output channel
                 M: integer in [16384, 32767] (15-bit unsigned), S: shift in [1, 47];
                 the product needs 48 bits; >> is an arithmetic shift (floor), so the
                 rounding is "half up" (towards +infinity)
  activation   ReLU layers clamp to [0, 127]; other layers clamp to [-128, 127]
  padding      zeros, k//2 on every side; Hout = (Hin + 2*pad - k) // stride + 1
  "fc" layer   global average pool, then fully connected:
                 g[c] = requant(sum over H*W of x[h,w,c])        (own M/S per channel, ReLU)
                 logits = requant(bias + W @ g)                  (no ReLU)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from tiny_skeleton import MODELS                                    # noqa: E402

INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
CONTRACT_VERSION = 1

def out_size(n, k, s):
    return (n + 2 * (k // 2) - k) // s + 1

def check_int32(acc, where):
    lo, hi = int(acc.min()), int(acc.max())
    assert INT32_MIN <= lo and hi <= INT32_MAX, f"int32 accumulator overflow in {where}: [{lo}, {hi}]"

def requant(acc, M, S, relu):
    """acc: int64 array (..., C); M, S: int arrays (C,).  Returns int8 array."""
    M = np.asarray(M, np.int64); S = np.asarray(S, np.int64)
    assert np.all((16384 <= M) & (M <= 32767)) and np.all((1 <= S) & (S <= 47))
    y = (acc.astype(np.int64) * M + (np.int64(1) << (S - 1))) >> S
    return np.clip(y, 0 if relu else -128, 127).astype(np.int8)

def windows(x, k, s):
    """Zero-pad x (H, W, C) by k//2 and yield (i, j, slab) for each kernel tap; slab is (Ho, Wo, C)."""
    H, W, _ = x.shape
    p = k // 2
    xp = np.pad(x.astype(np.int64), ((p, p), (p, p), (0, 0)))
    ho, wo = out_size(H, k, s), out_size(W, k, s)
    for i in range(k):
        for j in range(k):
            yield i, j, xp[i:i + s * (ho - 1) + 1:s, j:j + s * (wo - 1) + 1:s, :]

def conv_acc(x, w, b, k, s):      # w: (Cout, k, k, Cin)
    acc = None
    for i, j, slab in windows(x, k, s):
        t = slab @ w[:, i, j, :].astype(np.int64).T
        acc = t if acc is None else acc + t
    return acc + b

def dw_acc(x, w, b, k, s):        # w: (C, k, k)
    acc = None
    for i, j, slab in windows(x, k, s):
        t = slab * w[:, i, j].astype(np.int64)
        acc = t if acc is None else acc + t
    return acc + b

def pw_acc(x, w, b):              # w: (Cout, Cin)
    return x.astype(np.int64) @ w.astype(np.int64).T + b

def layer_forward(L, P, x):
    """One layer.  L: shape dict from tiny_skeleton; P: parameters.  Returns (acc, y) and,
    for "fc", the pooled vector too.  acc is the int32 value before requantisation."""
    kind = L["kind"]
    if kind == "conv":
        acc = conv_acc(x, P["w"], P["b"], L["k"], L["s"])
    elif kind == "dw":
        acc = dw_acc(x, P["w"], P["b"], L["k"], L["s"])
    elif kind == "pw":
        acc = pw_acc(x, P["w"], P["b"])
    else:                                                            # gap + fc
        pool_acc = x.astype(np.int64).sum(axis=(0, 1))
        check_int32(pool_acc, L["name"] + ".pool")
        g = requant(pool_acc, P["pool_M"], P["pool_S"], relu=True)
        acc = pw_acc(g[None, :], P["w"], P["b"])[0]
        check_int32(acc, L["name"])
        return acc, requant(acc, P["M"], P["S"], relu=False), (pool_acc, g)
    check_int32(acc, L["name"])
    return acc, requant(acc, P["M"], P["S"], relu=True), None

def forward(model, params, x):
    """Run a whole model.  Returns a list of per-layer records (acc, out, pool)."""
    layers, _ = MODELS[model]
    trace = []
    for L, P in zip(layers, params):
        acc, x, pool = layer_forward(L, P, x)
        trace.append(dict(acc=acc, out=x, pool=pool))
    return trace

def input_shape(model):
    L = MODELS[model][0][0]
    return (L["h"], L["wd"], L["ci"])

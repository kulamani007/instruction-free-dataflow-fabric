"""
Precision comparison: rounding error of INT8, INT4 and FP4 (E2M1) formats  (v1)

Quantises synthetic weight-like data and reports SQNR (signal-to-quantisation-
noise ratio, dB; higher is more precise; +6 dB is about one extra bit).

  INT8 / INT4 / FP4 per-tensor : one scale for the whole tensor
  INT4 block-32                : one scale per 32 values
  MXFP4 block-32               : OCP Microscaling FP4, one power-of-two (E8M0)
                                 scale per 32 values

NVFP4 and other vendor-specific formats are excluded by decision (see
notes/2026-11-01_precision-plan.md).  Data is synthetic (Gaussian, Laplacian,
Gaussian with rare outliers); accuracy on trained models needs a separate test.
"""
import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
GRID = np.concatenate([-E2M1[::-1], E2M1])

def fp4(x, scale):                       # round x/scale to the nearest E2M1 value
    y = np.clip(x / scale, -6, 6)
    return GRID[np.abs(y[..., None] - GRID).argmin(-1)] * scale

def intq(x, scale, qmax):
    return np.clip(np.round(x / scale), -qmax - 1, qmax) * scale

def blocks(x, n, f):
    return np.concatenate([f(r) for r in x.reshape(-1, n)]).reshape(x.shape)

def sqnr(x, q):
    return 10 * np.log10((x**2).sum() / ((x - q)**2).sum())

SCHEMES = {
    "INT8  per-tensor":          lambda x: intq(x, np.abs(x).max() / 127, 127),
    "INT4  per-tensor":          lambda x: intq(x, np.abs(x).max() / 7, 7),
    "FP4   per-tensor":          lambda x: fp4(x, np.abs(x).max() / 6),
    "INT4  block-32":            lambda x: blocks(x, 32, lambda r: intq(r, np.abs(r).max() / 7, 7)),
    "MXFP4 block-32 (2^k scale)": lambda x: blocks(x, 32, lambda r: fp4(
        r, 2.0 ** (np.floor(np.log2(np.abs(r).max())) - 2))),
}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"FP4 (E2M1) values: {sorted(set(GRID.tolist()))} -> {len(set(GRID.tolist()))} distinct;"
          f" INT8: 256 distinct\n")
    n = 1 << 15
    data = {"Gaussian": rng.normal(size=n),
            "Laplacian": rng.laplace(size=n),
            "Gaussian + 0.1% outliers x10": np.where(rng.random(n) < 1e-3, 10, 1) * rng.normal(size=n)}
    print(f"{'SQNR (dB)':<28}" + "".join(f"{k:>30}" for k in data))
    for name, f in SCHEMES.items():
        print(f"{name:<28}" + "".join(f"{sqnr(x, f(x)):>30.1f}" for x in data.values()))

    # FP4 x FP4 products are exact multiples of 0.25 in [-36, 36]: 9-bit integers after x4
    prods = {a * b * 4 for a in GRID for b in GRID}
    assert all(float(p).is_integer() for p in prods) and max(abs(p) for p in prods) == 144
    print("\nFP4 x FP4: 256-entry product table; every product x4 is an integer in [-144, 144]"
          " (9-bit signed), so accumulation can be integer")

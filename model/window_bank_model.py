"""
Window-extraction model for a processor-free dataflow stage  (v2)

Part A - Address generator + banked fetch (functional check)
   A base-address generator walks the window's top-left pixel (r0, c0) by
   the stride.  Pixel (r,c) lives in bank (r % Kr, c % Kc) at local address
   (r//Kr)*LW + c//Kc.  Every bank is read at most once per window, so a full
   MxN window arrives in ONE cycle.  Checked bit-exact against numpy slicing.

Part B - Fetch modes (SRAM-read counting)
   FULL  : re-fetch all M*N pixels for every window (any stride, any order)
   SHIFT : keep the window in a register; on a horizontal step of s, fetch
           only the min(s, N) new columns (M*min(s,N) pixels). First window of
           each row is fetched in full.  Also checked bit-exact.
   Run on the 3x3 layers of MobileNetV1 (padding 1) to count SRAM reads.
"""
import numpy as np, itertools

# ---------------------------------------------------------------- Part A ----
def addr_gen(H, W, M, N, s):
    """Base-address generator: yields (r0, c0) in row-major order."""
    r0 = 0
    while r0 + M <= H:
        c0 = 0
        while c0 + N <= W:
            yield r0, c0
            c0 += s                      # slide by stride
        r0 += s                          # wrap to next window row

def build_banks(img, Kr, Kc):
    H, W = img.shape
    LW = -(-W // Kc)
    banks = {(i, j): {} for i in range(Kr) for j in range(Kc)}
    for r, c in itertools.product(range(H), range(W)):
        banks[(r % Kr, c % Kc)][(r // Kr) * LW + c // Kc] = img[r, c]
    return banks, LW

def fetch_pixel(banks, LW, Kr, Kc, r0, c0, r, c):
    b = (r % Kr, c % Kc)
    ar = r0 // Kr + (1 if b[0] < r0 % Kr else 0)   # base + carry
    ac = c0 // Kc + (1 if b[1] < c0 % Kc else 0)
    return b, banks[b][ar * LW + ac]

def check_banked(H, W, M, N, s, Kr, Kc):
    img = np.random.randint(0, 256, (H, W))
    banks, LW = build_banks(img, Kr, Kc)
    n = 0
    for r0, c0 in addr_gen(H, W, M, N, s):
        used, win = set(), np.zeros((M, N), int)
        for dr, dc in itertools.product(range(M), range(N)):
            b, v = fetch_pixel(banks, LW, Kr, Kc, r0, c0, r0 + dr, c0 + dc)
            assert b not in used, "bank conflict"
            used.add(b); win[dr, dc] = v
        assert (win == img[r0:r0+M, c0:c0+N]).all()
        n += 1
    return n, M * N / (Kr * Kc)

# ---------------------------------------------------------------- Part B ----
def reads(H, W, M, N, s, mode, verify=False):
    img = np.random.randint(0, 256, (H, W)) if verify else None
    total, reg, prev_r = 0, None, None
    for r0, c0 in addr_gen(H, W, M, N, s):
        if mode == "FULL" or r0 != prev_r:      # full fetch
            total += M * N
            if verify: reg = img[r0:r0+M, c0:c0+N].copy()
        else:                                   # shift in new columns only
            k = min(s, N)
            total += M * k
            if verify:
                new = img[r0:r0+M, c0+N-k:c0+N]
                reg = np.concatenate([reg[:, k:], new], axis=1) if k < N else new
        prev_r = r0
        if verify: assert (reg == img[r0:r0+M, c0:c0+N]).all()
    return total

if __name__ == "__main__":
    np.random.seed(0)
    print("Part A: address generator + banked single-cycle window fetch")
    for (H,W,M,N,s,Kr,Kc) in [(8,8,3,3,1,3,3),(8,8,3,3,2,3,3),(58,58,3,3,2,3,3),
                              (16,16,3,3,1,5,5),(16,16,5,5,1,5,5),(16,16,1,1,1,5,5),
                              (15,20,3,5,2,3,5)]:
        n, a = check_banked(H,W,M,N,s,Kr,Kc)
        print(f"  img {H:>2}x{W:<2} win {M}x{N} s{s} banks {Kr}x{Kc}: "
              f"{n:>4} windows bit-exact, 0 conflicts, banks active {a:4.0%}")

    print("\nPart B: SHIFT mode functional check")
    for s in (1, 2, 3):
        reads(20, 20, 3, 3, s, "SHIFT", verify=True)
        print(f"  3x3 stride {s}: bit-exact")

    # MobileNetV1 3x3 layers: (name, input HxW, channels, stride); padding 1
    net = [("conv1",224,3,2),("dw1",112,32,1),("dw2",112,64,2),("dw3",56,128,1),
           ("dw4",56,128,2),("dw5",28,256,1),("dw6",28,256,2)] + \
          [(f"dw{i}",14,512,1) for i in range(7,12)] + \
          [("dw12",14,512,2),("dw13",7,1024,1)]
    print("\nPart B: SRAM reads, MobileNetV1 3x3 layers (padding 1)")
    print(f"  {'layer':<6}{'in':>5}{'ch':>6}{'s':>3}{'FULL (M)':>11}{'SHIFT (M)':>11}{'saving':>8}")
    tf = ts = 0
    by_s = {1: [0, 0], 2: [0, 0]}
    for name, hw, ch, s in net:
        P = hw + 2
        f = reads(P, P, 3, 3, s, "FULL") * ch
        sh = reads(P, P, 3, 3, s, "SHIFT") * ch
        tf += f; ts += sh; by_s[s][0] += f; by_s[s][1] += sh
        print(f"  {name:<6}{hw:>5}{ch:>6}{s:>3}{f/1e6:>11.2f}{sh/1e6:>11.2f}{1-sh/f:>8.0%}")
    print(f"  {'TOTAL':<20}{tf/1e6:>11.2f}{ts/1e6:>11.2f}{1-ts/tf:>8.0%}")
    for s,(f,sh) in by_s.items():
        print(f"  stride-{s} layers only: SHIFT saves {1-sh/f:.0%} of reads")

    print("\nPart B: where reuse disappears (stride >= kernel width)")
    for (M,N,s,lbl) in [(3,3,1,"3x3 conv s1"),(3,3,2,"3x3 conv s2"),(2,2,2,"2x2 pool s2"),
                        (3,3,3,"3x3 s3"),(1,1,1,"1x1 conv s1")]:
        f = reads(58,58,M,N,s,"FULL"); sh = reads(58,58,M,N,s,"SHIFT")
        print(f"  {lbl:<12}: SHIFT saves {1-sh/f:4.0%}")

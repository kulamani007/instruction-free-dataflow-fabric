# Golden model for the RTL (hardware step H0)

The RTL must match these files **bit for bit**. Everything here is generated from the three
tiny-skeleton CNNs defined in [`model/tiny_skeleton.py`](../model/tiny_skeleton.py).

```bash
python golden/make_vectors.py      # writes golden/vectors/, checks vectors.sha256
```

```bash
python golden/test_golden.py       # self-tests against independent references
```

`golden/vectors/` is not committed (it regenerates identically in under a second);
`golden/vectors.sha256` is, so any change to the numbers is detected.

## Arithmetic contract (version 1)

Full text in [`golden_model.py`](golden_model.py). In short:

| Item | Rule |
|---|---|
| Activations | int8, zero point 0, layout H × W × C (channel fastest, raster order) |
| Weights | int8 per output channel; conv `[Cout][kh][kw][Cin]`, dw `[C][kh][kw]`, pw / fc `[Cout][Cin]` |
| Bias | int32 per output channel, added to the accumulator |
| Requantise | `y = (acc * M + 2^(S-1)) >> S`, arithmetic shift; M ∈ [16384, 32767], S ∈ [1, 47]; 48-bit product |
| Activation | ReLU: clamp [0, 127]; logits: clamp [−128, 127] |
| Padding | zeros, k//2 each side; `Hout = (Hin + 2·pad − k) // stride + 1` |
| "fc" layer | global sum over H×W → requantise (own M/S, ReLU) → FC → requantise (no ReLU) |

## Accumulator width

Measured on the test vectors and guaranteed for any int8 input:

| Model | Bits seen | Worst case |
|---|---|---|
| VisionNet | 18 | 25 (c5: 576 products) |
| KWSNet | 17 | 21 |
| PersonNet | 16 | 21 |

A 25-bit signed accumulator is enough for all three models; 32 bits is not needed.

## Files per model (`golden/vectors/<model>/`)

| File | Content | Hex width |
|---|---|---|
| `config.json` | layer list: kind, k, stride, pad, input/output H×W×C, ReLU, accumulator widths | n/a |
| `input_<n>.hex` | test input n | 2 |
| `<layer>.weights.hex`, `.bias.hex` | parameters | 2, 8 |
| `<layer>.M.hex`, `.S.hex` | requantisation, one line per output channel | 4, 2 |
| `<layer>.acc_<n>.hex` | int32 accumulator before requantisation (checks the MAC array alone) | 8 |
| `<layer>.out_<n>.hex` | int8 layer output (for `fc`: the logits) | 2 |
| `fc.pool_acc_<n>.hex`, `fc.pool_<n>.hex`, `fc.pool_M.hex`, `fc.pool_S.hex` | pooling stage of `fc` | 8, 2, 4, 2 |

One value per line, two's complement, lower-case hex: ready for `$readmemh`. Weights are
random (seeded) and calibrated so activations use the int8 range; they are not trained, which
does not matter for bit-exactness.

## Suggested verification order

1. **H1 window unit:** compare fetched windows with `model/window_bank_model.py`
   (test 3 of `test_golden.py` already shows that banked fetch + MAC reproduces `c1.acc_*`).
2. **H2 MAC array:** compare against `<layer>.acc_<n>.hex` first, then requantisation against
   `<layer>.out_<n>.hex`; a mismatch then points at one block.
3. **H3** depthwise, pointwise and `fc` modes, the same way.
4. **H4** KWSNet end to end: final `fc.out_<n>.hex`.

The RTL itself stays out of the public repository until the patent decision.

# Idea: replace multiplication with lookup tables (LUT storage of precomputed results)

**Author:** Kulamani Rout · **First recorded:** 1 November 2026 · **Status:** idea, assessed; not adopted for the core fabric yet

## The idea

Skip computation wherever possible. For multiplication and similar operations, store every
possible result in a table in advance; at run time, look the answer up instead of
computing it. The aim is to avoid the cost of DSP-style arithmetic.

## Assessment

**Full INT8 tables do not pay off.**

| Operand widths | Table entries | Result width | Size of one table |
|---|---|---|---|
| INT8 × INT8 | 65,536 | 16 bit | 128 KB |
| INT4 × INT4 | 256 | 8 bit | 256 B |
| INT2 × INT2 | 16 | 4 bit | 8 B |

- **Size:** each MAC needs its own table, because one memory port gives one lookup per
  cycle. The 320-MAC tiny skeleton would need 160–320 copies of the INT8 table (20–40 MB),
  far more than the ZU19EG's ~8.8 MB of on-chip RAM; a multiplier is a few hundred gates.
- **Speed:** a pipelined multiplier (or an FPGA DSP slice) already returns one product per
  cycle, and a RAM lookup also takes a cycle or two. There is no time saving at INT8.
- **Energy:** reading SRAM costs far more than an 8-bit multiply. Horowitz (ISSCC 2014,
  45 nm) gives about 0.2 pJ for an 8-bit multiply and about 10 pJ to read 64 bits from an
  8 KB SRAM. Data movement, not arithmetic, is the dominant cost in accelerators.

**Where the idea is right, and is an active research line:**

1. **Low precision (≤ 4-bit weights).** Tables become tiny (256 B at INT4). T-MAC (Microsoft
   Research, 2024) runs low-bit LLM matmuls on CPUs as table lookup + add with no
   multiplications; LUT Tensor Core (ISCA 2025) builds this into hardware. The trick is
   to precompute sums of groups of activations and index them with the weight bits.
2. **The neuron as a table.** LUTNet (FCCM 2019) and LogicNets (Umuroglu et al., FPL 2020)
   turn whole low-bit neurons into truth tables on FPGA LUTs: input bits go in, the output
   comes out, with no arithmetic at all. This is the purest form of "the model as a
   circuit", and close in spirit to the skeleton. In fixed silicon the tables would live in
   SRAM and be loaded as configuration.
3. **Fixed weights help.** In the skeleton, weights are stationary configuration, so tables
   that depend on the weights can be precomputed by the compiler at load time.
4. **Activation functions** are already planned as LUTs (sigmoid, SiLU, hardswish).

## Decision for now

Keep INT8 multipliers in the core engines. Record a **low-bit LUT engine** (INT4 / ternary
weights, table lookup + add) as a candidate engine type for the efficiency question (RQ2),
to be evaluated once the base skeleton works. It is not part of the first prototype, to
keep the PhD's first contribution focused.

## Sources

- T-MAC: <https://www.microsoft.com/en-us/research/?p=1112340>, <https://arxiv.org/abs/2407.00088v1>
- LUT Tensor Core: <https://arxiv.org/html/2408.06003v2>
- LUTNet: <https://talks.ee.ic.ac.uk/talk/index/1116>
- M. Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014.
- Y. Umuroglu et al., "LogicNets: Co-Designed Neural Networks and Circuits for Extreme-Throughput Applications," FPL 2020.

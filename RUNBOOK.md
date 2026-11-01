# Runbook

Every run and test performed on this project: the exact commands, in the order they were
run, and what came out. New runs are **appended at the bottom of the run log** with the
date, so the file reads as a history.

**Rules for updating**
- One entry per date (or per session); inside it, number the runs R1, R2, … in the order
  they were done. A run is referred to as `YYYY-MM-DD-Rn`.
- Write the exact commands (from the repo root), the key result numbers, the commit that
  holds the code or result, and anything that went wrong and how it was fixed.
- Never rewrite an old entry to match new results. If something is re-run with a different
  outcome, add a new entry and say what changed.
- When a run's key numbers become a reference, add them to `tools/check_all.sh`.

---

## 1. One-time setup

**Mac (current machine)**: repo at `~/Desktop/Claude-Code-things/instruction-free-dataflow-fabric`.

```bash
cd ~/Desktop/Claude-Code-things/instruction-free-dataflow-fabric
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install numpy==1.26.4 matplotlib==3.9.4
```

Only needed for specific runs:

| Purpose | Install |
|---|---|
| Re-extract CNN layer tables (`model/workloads/extract_layers.py`) | `pip install torch==2.8.0 torchvision==0.23.0 ultralytics==8.4.197` |
| PDF inspection or patching (`tools/patch_proposal_pdf.py`) | `pip install pymupdf==1.26.5 fonttools==4.60.2` |

GitHub push access (once per machine):

```bash
brew install gh
```

```bash
gh auth login
```

```bash
gh auth setup-git
```

Reference environment for all 1 Nov 2026 runs: macOS 26.6 (arm64), Python 3.9.6,
numpy 1.26.4, matplotlib 3.9.4, PyMuPDF 1.26.5, fonttools 4.60.2, torch 2.8.0,
torchvision 0.23.0, ultralytics 8.4.197.

## 2. Regression check (run after any change)

```bash
bash tools/check_all.sh
```

It re-runs every model, checks the key numbers below, re-draws the figures and reports
whether they changed. Expected last line: `ALL CHECKS PASSED`.

| Script | Reference numbers |
|---|---|
| `model/window_bank_model.py` | FULL 17.72 M, SHIFT 7.56 M reads, 57 % saving; stride-2 layers 32 % |
| `model/composability_model.py` | DS-CNN worst case on time-shared NPU 6.404 ms (129.23×), 12.361 ms (249.42×) with two cameras; MobileNetV1 fixed 18.653 ms (P-fair) |
| `model/fit_map.py` | ResNet-50 on Large: circuit, 28 % MACs busy; MobileNetV1 on Medium: circuit, 52 %; VGG-16 on Large: streamed, 9 % |
| `model/tiny_skeleton.py` | 20 engines, 1280 KB weight SRAM 6 % used; M = 16 → 320 DSPs (16 % of ZU19EG); 8 KB engines → 27 engines, 216 KB |
| `model/precision_compare.py` | INT8 SQNR 39.8 / 35.3 / 26.1 dB; MXFP4 18.8 / 17.9 / 18.4 dB |

## 3. Run log

### 2026-09-23 · Windows laptop (SHANI007) · window-extraction model

*Reconstructed from the first session's handoff; exact command order not recorded.*

- R1 Built `model/window_bank_model.py` v2 (address generator, banked fetch, FULL/SHIFT).
  ```bash
  python model/window_bank_model.py
  ```
  Result: bit-exact, conflict-free window fetch for 3×3, 5×5, 3×5, 1×1 at stride 1 and 2;
  MobileNetV1 3×3 layers: FULL 17.7 M → SHIFT 7.6 M reads (57 %).
  Corrections made then: stride-1 FULL total is 14.0 M (not 13.0 M); SHIFT never loses on
  reads, saving is 0 % only when stride ≥ kernel width.

### 2026-09-28 / 29 · Windows laptop · proposal PDF build and first public commits

*Reconstructed from the handoff.*

- R1 Proposal PDF built from `proposal/Research_Proposal_Kulamani_Rout.md`: pandoc
  Markdown → HTML; SVG Figure 1 injected at the `<!--FIGURE-->` marker; references wrapped in
  `<div class="refs">`; `<colgroup>` removed; page break before §3.2; wkhtmltopdf, A4,
  margins 18/18/20/20 mm; fonts TeX Gyre Termes (body) and Heros (headings). The Figure 1
  SVG/CSS lived in a temporary folder and is not in the repo.
- R2 Commits `ec3aab2` (28 Sep, initial public record) and `9713439` (29 Sep, validation
  method), pushed over HTTPS through `gh`.

### 2026-09-30 · Windows laptop · polyphase check

- R1 Polyphase (space-to-depth) stride-2 convolution checked bit-exact against direct
  convolution; code in [notes/2026-09-30_external-memory-and-polyphase.md](notes/2026-09-30_external-memory-and-polyphase.md).
  im2col storage overhead: 2.21× (3×3, stride 2, 112×112), 8.39× (3×3, stride 1, 56×56).

### 2026-11-01 · Mac (Claude Code desktop) · second session

- **R1 Environment.** System Python 3.9.6 had no numpy; created a virtual environment and
  installed numpy (then matplotlib, PyMuPDF, fonttools, torch, torchvision, ultralytics as
  later runs needed them; versions in section 1). That day's runs used a temporary venv;
  the repo's `.venv` (section 1) was created at the end of the day to verify this runbook.

- **R2 Re-run window model (verification).**
  ```bash
  python model/window_bank_model.py
  ```
  Matched the 23 Sep results exactly (17.72 M / 7.56 M / 57 %).

- **R3 Proposal PDF patch (email + DLA citation).** No pandoc, wkhtmltopdf or Figure 1
  source on the Mac, so the existing PDF was edited in place with its own embedded fonts.
  ```bash
  python tools/patch_proposal_pdf.py proposal/Research_Proposal_Kulamani_Rout.pdf /tmp/patched.pdf
  ```
  Issues fixed on the way: ToUnicode map used the array form of `bfrange`; the space glyph
  was mapped to U+0009. Checks: text diff showed exactly the two edits; page 2 pixel-identical;
  pages 1 and 3 inspected as images. Copied over the proposal PDF; commit `1a2d81e`.
  (One-off: re-running fails because the old email is gone.)

- **R4 Composability model, first version.** Three models (MobileNetV1, DS-CNN, KWT-1) on
  256 MACs / 3.2 GB/s; systems B1-FIFO, B1-RM, B2, P; load swept ×1, ×2, ×3.
  ```bash
  python model/composability_model.py
  ```
  Findings that changed the design: the deadline-only compiler objective gave KWT-1 1/32 of
  the array (18.4 ms guaranteed, worse than the NPU's 7.2 ms worst case); the load sweep
  added nothing (worst case is set by blocking, not load); DRAM interference was tiny for
  this mix. This version was revised in place before committing (R5).

- **R5 Composability model, committed version.** Added the "fair" compiler objective
  (P-fair) and a DRAM-heavy scenario (two MobileNetV1 cameras + DS-CNN); dropped the sweep.
  ```bash
  python model/composability_model.py
  ```
  Scenario A worst cases: DS-CNN 0.050 ms alone → 6.40 ms co-running on B1 (129×), P-fair
  fixed 0.141 ms; MobileNetV1 P-fair 18.65 ms (2.9× its full-chip time). Scenario B: DS-CNN
  249× on B1. Runs in ~3 s.

- **R6 Composability figure.**
  ```bash
  python model/plot_composability.py
  ```
  → `figures/composability.png`. Iteration: removed stray minor ticks on the log axis.
  Commit `4d85011`.

- **R7 Extract layer tables of 11 CNNs.**
  ```bash
  pip install torch==2.8.0 torchvision==0.23.0 ultralytics==8.4.197
  ```
  ```bash
  YOLO_OFFLINE=1 python model/workloads/extract_layers.py
  ```
  First run problems, all fixed before committing: ReLU6 not detected (torchvision uses
  `hardtanh`); YOLOv8 box decoding traced because `.eval()` reset the Detect head
  (now forced to raw-output mode after `.eval()`); VGG's 7×7 adaptive pool mislabelled as
  global. Totals match published figures (e.g. ResNet-50 25.56 M params / 4.09 GMAC,
  YOLOv8n 3.16 M / 4.37 GMAC). Output: `model/workloads/*.csv`, `model/workloads/ops.json`.

- **R8 Skeleton fit map.**
  ```bash
  python model/fit_map.py
  ```
  Then added the "folding instead of circuit" comparison and re-ran. Results in
  [notes/2026-11-01_composability-and-fit-map-results.md](notes/2026-11-01_composability-and-fit-map-results.md).

- **R9 Fit map figure.**
  ```bash
  python model/plot_fit_map.py
  ```
  → `figures/fit_map.png`. Iterations: header overlap fixed, rows sorted by weights,
  note added that folded utilisation is idealised.

- **R10 Re-verify the 30 Sep polyphase check and im2col ratios.** Ran the note's code block
  5 times (random inputs): all bit-exact. im2col ratios reproduced (2.21×, 8.39×) with
  TensorFlow-style "same" padding.

- **R11 Smoke test of every README command.** All five scripts ran; `figures/composability.png`
  regenerated byte-identical. Commit `3883332`.

- **R12 Tiny skeleton v0.**
  ```bash
  python model/tiny_skeleton.py
  ```
  Then added the engine-SRAM trade-off table and re-ran. 20 engines, ZU19EG 16 % DSP /
  32 % BRAM at 16 MACs per engine; time-shared NPU 20× faster per model at this size.
  Commit `887062d`.

- **R13 LUT table sizes** (inline calculation): INT8×INT8 table 128 KB, INT4×INT4 256 B.
  Recorded in [notes/2026-11-01_lut-multiplication-idea.md](notes/2026-11-01_lut-multiplication-idea.md).

- **R14 Precision comparison.**
  ```bash
  python model/precision_compare.py
  ```
  INT8 ≈ 3 bits more precise than any 4-bit format; FP4 × FP4 products fit 9-bit integers
  (asserted in the script). NVFP4 excluded by decision. Commits `16820a5`, `ea541fd`.

- **R15 Download the MX paper** (Rouhani et al., arXiv 2310.10537 v3) to `papers/`, which is
  excluded from git (`.git/info/exclude`): its arXiv licence does not allow redistribution.
  ```bash
  curl -sSL -o papers/Rouhani2023_Microscaling_Data_Formats_for_Deep_Learning.pdf https://arxiv.org/pdf/2310.10537v3
  ```

- **R16 Runbook verification.** Created the repo's `.venv` exactly as in section 1 and ran
  the regression check.
  ```bash
  bash tools/check_all.sh
  ```
  Result: ALL CHECKS PASSED (16 checks) in a fresh `.venv` with numpy 1.26.4 and matplotlib 3.9.4; figures byte-identical. A forced failure (`PYTHON=false`) is reported as FAIL.

---

## 4. Planned runs

From the notes, in rough priority. Move each one into the run log when it is done.

1. Skip-connection buffer sizing in the fit map (YOLO long skips).
2. Time-triggered static schedule as a third option in the composability model.
3. Work-conserving DRAM with a guaranteed minimum share; then a DRAM timing model
   (Ramulator / DRAMSys) for bound tightness (claim C2).
4. Per-operator array utilisation (depthwise on a dense array) in both models.
5. Polyphase mode in `model/window_bank_model.py`; compare SRAM/DRAM reads with FULL and SHIFT.
6. Train and quantise the three tiny-skeleton models (CIFAR-10, Speech Commands, visual wake
   words); INT8 vs MXFP4 vs block INT4 accuracy.
7. RTL for one engine (window unit + MAC array + weight SRAM), checked against the Python
   model with Verilator; then the full tiny skeleton on the ZU19EG.

## 5. Template for a new entry

````markdown
### YYYY-MM-DD · <machine> · <topic>

- **R1 <what and why>.**
  ```bash
  <exact command>
  ```
  Result: <key numbers>. Problems and fixes: <...>. Commit `<hash>`.
````

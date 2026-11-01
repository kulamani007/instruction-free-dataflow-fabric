"""
Figure for the composability experiment: worst-case slowdown of each model when
co-running, relative to running alone on the full chip (log scale).

  python model/plot_composability.py      ->  figures/composability.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from composability_model import SCENARIOS, evaluate, full_chip

SERIES = [("B1-RM", "Time-shared NPU (priority)", "#2a78d6"),
          ("B2",    "Split compute, shared DRAM", "#eb6834"),
          ("P-fair", "Proposed: static compute + DRAM shares", "#1baf7a")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "Helvetica", "font.size": 8.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, facecolor=SURFACE,
                         gridspec_kw={"width_ratios": [1, 1]})
w, gap = 0.26, 0.02
for ax, (title, models) in zip(axes, SCENARIOS):
    _, res = evaluate(models, trials=40)
    ax.set_facecolor(SURFACE)
    for k, (label, legend, color) in enumerate(SERIES):
        xs, ys = [], []
        for i, (name, L, _) in enumerate(models):
            xs.append(i + (k - 1) * (w + gap))
            ys.append(res[(name, label)]["max"] / full_chip(L))
        ax.bar(xs, ys, width=w, color=color, label=legend, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.12, f"{y:.0f}×" if y >= 10 else f"{y:.1f}×", ha="center",
                    va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 900)
    ax.axhline(1, color=INK2, lw=0.8, zorder=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([n for n, _, _ in models])
    ax.tick_params(which="both", length=0)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Scenario {title}", fontsize=8.5, color=INK, loc="left")
axes[0].set_ylabel("Worst-case latency ÷ latency alone on full chip")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=8, labelcolor=INK)
fig.text(0.01, -0.02, "B1, B2: worst latency observed over 40 random-phase runs of 1 s.  "
         "Proposed: compile-time bound (identical alone or co-running).  Same 256 MACs, 3.2 GB/s DRAM.",
         fontsize=6.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = os.path.join(os.path.dirname(__file__), "..", "figures", "composability.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=SURFACE)
print("wrote", os.path.normpath(out))

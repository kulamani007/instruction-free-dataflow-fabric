"""
Figure for the skeleton fit map: for each CNN and skeleton size, what the model
becomes (circuit / folded / streamed) and how well it fills the MACs.

  python model/plot_fit_map.py      ->  figures/fit_map.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from fit_map import SKELETONS, ORDER, F_CLK, fit_all

FILL = {"CIRCUIT": ("#1c5cab", "#ffffff"), "FOLDED": ("#6da7ec", "#0b0b0b"),
        "STREAMED": ("#cde2fb", "#0b0b0b")}
LABEL = {"CIRCUIT": "Circuit", "FOLDED": "Folded", "STREAMED": "Streams weights"}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

info, res = fit_all()
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8})
fig, ax = plt.subplots(figsize=(8.6, 4.9), facecolor=SURFACE)
ax.set_facecolor(SURFACE); ax.axis("off")
n = len(ORDER)
x_model, x_w, x0, cw, x_ops = 0.0, 1.85, 2.55, 1.75, 7.9
ax.set_xlim(-0.05, 10.3); ax.set_ylim(n + 1.9, -0.45)
ORDER = sorted(ORDER, key=lambda m: res[(m, SKELETONS[0][0])]["weights"])

hdr = dict(fontsize=7.5, color=INK2, va="center")
ax.text(x_model, 0.35, "Model", **hdr)
ax.text(x_w + 0.45, 0.35, "Weights", ha="right", **hdr)
for k, (name, E, mpe, sram, bw) in enumerate(SKELETONS):
    cx = x0 + k * cw + cw / 2
    ax.text(cx, -0.1, f"{name} skeleton", ha="center", fontsize=8, color=INK, weight="bold", va="center")
    ax.text(cx, 0.4, f"{E} engines\n{E*sram/2**20:.0f} MB SRAM · {2*E*mpe*F_CLK/1e12:.0f} TOPS",
            ha="center", fontsize=6.3, color=INK2, va="center", linespacing=1.2)
ax.text(x_ops, 0.35, "Operators to add", **hdr)

for i, m in enumerate(ORDER):
    y = i + 1.25
    r0 = res[(m, SKELETONS[0][0])]
    ax.text(x_model, y, m, fontsize=8, color=INK, va="center")
    w = r0["weights"] / 1e6
    ax.text(x_w + 0.45, y, f"{w:.2f} MB" if w < 10 else f"{w:.0f} MB", ha="right",
            fontsize=7.5, color=INK2, va="center")
    for k, s in enumerate(SKELETONS):
        r = res[(m, s[0])]
        fill, ink = FILL[r["cls"]]
        ax.add_patch(FancyBboxPatch((x0 + k * cw + 0.04, y - 0.42), cw - 0.08, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=fill, edgecolor=SURFACE, lw=1.5))
        ax.text(x0 + k * cw + cw / 2, y - 0.14, LABEL[r["cls"]], ha="center", va="center",
                fontsize=7.2, color=ink, fontweight="semibold")
        ax.text(x0 + k * cw + cw / 2, y + 0.2, f"{r['util']:.0%} of MACs busy", ha="center",
                va="center", fontsize=6, color=ink)
    ext = r0["ext"]
    ax.text(x_ops, y, "\n".join(ext) if ext else "none", fontsize=6.5,
            color=INK if ext else INK2, va="center", linespacing=1.15)

handles = [Patch(facecolor=FILL[c][0], label=t) for c, t in
           [("CIRCUIT", "Circuit: every layer has its own engine, weights stay on-chip"),
            ("FOLDED", "Folded: weights on-chip, engines shared by layers on a fixed schedule"),
            ("STREAMED", "Streams weights from DRAM every frame")]]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.035), frameon=False,
          fontsize=6.8, ncol=1, handlelength=1.2, labelcolor=INK)
ax.text(6.0, n + 1.5, "All three run with no processor and no instructions; only the cost changes.\n"
        "Folded utilisation is idealised (switching engines between layers is free here).\n"
        "Operators checked against the proposal's op set and a 5×5 window unit.",
        fontsize=6.3, color=INK2, va="center", linespacing=1.3)
out = os.path.join(os.path.dirname(__file__), "..", "figures", "fit_map.png")
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=SURFACE)
print("wrote", os.path.normpath(out))

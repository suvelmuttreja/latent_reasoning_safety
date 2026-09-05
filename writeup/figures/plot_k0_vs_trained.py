"""Coconut GSM8K-200 accuracy with the latents on (trained K) versus off (K=0)
at fixed weights, per stage. Numbers are the frozen values from
writeup/claims_and_numbers.md (Coconut capability/coherence trajectory table);
the stage-3 K=0 entry uses the audited 48-58% no-imputation bound and the
49.5% cutoff-parser figure rather than the older 49.5-59.5% artifact bound.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLUE = "#4C78A8"
ORANGE = "#E45756"
GREY = "#777777"

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555555",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
    }
)

stages = ["u1\n(trained K=2)", "u2\n(trained K=4)", "u3\n(trained K=6)"]
trained = [65.5, 46.0, 31.0]
k0_point = [76.0, 69.5, None]  # stage 3 is a bound, not a point
k0_bound = (48.0, 58.0)
k0_parser = 49.5
k0_stops = ["8/200 unfinished", "5/200 unfinished", "20/200 unfinished\n(19 exact loops)"]
trained_stops = ["0/200", "0/200", "0/200"]

x = np.arange(3)
w = 0.34
fig, ax = plt.subplots(figsize=(6.7, 3.1))

# K=0 bars
for i in range(3):
    if k0_point[i] is not None:
        ax.bar(x[i] - w / 2, k0_point[i], width=w, color=GREY, label="Latents off (K=0)" if i == 0 else None)
        ax.text(x[i] - w / 2, k0_point[i] + 1.5, f"{k0_point[i]:.1f}%", ha="center", va="bottom", fontsize=9, color=GREY)
    else:
        lo, hi = k0_bound
        ax.bar(x[i] - w / 2, lo, width=w, color=GREY, alpha=0.35)
        ax.bar(x[i] - w / 2, hi - lo, bottom=lo, width=w, color="none", edgecolor=GREY, hatch="///", linewidth=1.0)
        ax.plot([x[i] - w / 2], [k0_parser], marker="o", color=GREY, markersize=4)
        ax.text(x[i] - w / 2, hi + 1.5, f"{lo:.0f}–{hi:.0f}%\n(49.5% parser)", ha="center", va="bottom", fontsize=8.5, color=GREY)
    ax.text(x[i] - w / 2, -9, k0_stops[i], ha="center", va="top", fontsize=7.5, color="#8B0000")

# trained-K bars
ax.bar(x + w / 2, trained, width=w, color=ORANGE, label="Latents on (trained K)")
for i in range(3):
    ax.text(x[i] + w / 2, trained[i] + 1.5, f"{trained[i]:.1f}%", ha="center", va="bottom", fontsize=9, color=ORANGE)
    ax.text(x[i] + w / 2, -9, trained_stops[i], ha="center", va="top", fontsize=7.5, color="#8B0000")

ax.axhline(92.0, color=BLUE, linewidth=1.0, linestyle="--")
ax.text(-0.45, 93.5, "Explicit CoT u3: 92.0%", ha="left", va="bottom", fontsize=8.5, color=BLUE)

ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylim(0, 105)
ax.set_ylabel("GSM8K-200 exact match")
ax.set_title("Coconut checkpoints with the latents on versus off (same weights)", loc="left", pad=26)
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8.5, ncol=2)
ax.tick_params(axis="x", pad=26)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.3, top=0.80)
out = ROOT / "writeup/figures/fig4_k0_vs_trained.png"
fig.savefig(out, dpi=220)
print("wrote", out.name)

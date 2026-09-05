"""Plot the matched capability trajectories from committed result summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts/discovery/results"

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

m0 = json.load(open(RESULTS / "native_gsm8k_controls/m0/summary.json"))
cot_summaries = [
    json.load(open(RESULTS / f"native_gsm8k_controls/cot_u{stage}/summary.json"))
    for stage in (1, 2, 3)
]
trajectory = json.load(
    open(RESULTS / "fallback_4b_skip0/k_trajectory_consolidated.json")
)["gsm8k_200"]

m0_lo, m0_hi = [100 * x for x in m0["accuracy_bounds_without_imputation"]]
m0_mid = (m0_lo + m0_hi) / 2
cot_bounds = [
    [100 * x for x in summary["accuracy_bounds_without_imputation"]]
    for summary in cot_summaries
]
cot_acc = [(lo + hi) / 2 for lo, hi in cot_bounds]
coconut_acc = [
    100 * trajectory["stage1"]["k2"]["accuracy"],
    100 * trajectory["stage2"]["k4"]["accuracy"],
    100 * trajectory["stage3"]["k6"]["accuracy"],
]

m0_x = 0.0
stage_x = [1.4, 2.6, 3.8]
bar_width = 0.38

fig, ax = plt.subplots(figsize=(7.4, 3.65))
m0_bar = ax.bar(m0_x, m0_mid, color=GREY, width=0.58, label="M0")
cot_bars = ax.bar(
    [x - bar_width / 2 for x in stage_x],
    cot_acc,
    color=BLUE,
    width=bar_width,
    label="Explicit CoT",
)
coconut_bars = ax.bar(
    [x + bar_width / 2 for x in stage_x],
    coconut_acc,
    color=ORANGE,
    width=bar_width,
    label="Coconut (K=2/4/6)",
)
ax.errorbar(
    m0_x,
    m0_mid,
    yerr=[[m0_mid - m0_lo], [m0_hi - m0_mid]],
    fmt="none",
    color="#333333",
    capsize=4,
    linewidth=1.2,
)
ax.set_title("Capability through training", loc="left")
ax.set_ylabel("GSM8K-200 accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-0.55, 4.35)
ax.set_xticks([m0_x, *stage_x])
ax.set_xticklabels(["M0", "u1", "u2", "u3"])
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01))

ax.text(
    m0_bar[0].get_x() + m0_bar[0].get_width() / 2,
    m0_mid + 2.0,
    f"{m0_lo:.1f}–{m0_hi:.1f}",
    ha="center",
    va="bottom",
    color=GREY,
    fontsize=9,
)
for bars, values, color, bounds in (
    (cot_bars, cot_acc, BLUE, cot_bounds),
    (coconut_bars, coconut_acc, ORANGE, None),
):
    for index, (bar, value) in enumerate(zip(bars, values)):
        label = f"{value:.1f}"
        if bounds is not None and bounds[index][0] != bounds[index][1]:
            label = f"{bounds[index][0]:.1f}–{bounds[index][1]:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
        )

for bar, value, (lo, hi) in zip(cot_bars, cot_acc, cot_bounds):
    if lo != hi:
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt="none",
            color="#333333",
            capsize=3,
            linewidth=1.0,
        )

fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig1_capability.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote fig1_capability.png")

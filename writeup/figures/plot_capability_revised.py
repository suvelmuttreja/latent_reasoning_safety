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
trajectory = json.load(open(RESULTS / "fallback_4b_skip0/k_trajectory_consolidated.json"))[
    "gsm8k_200"
]

m0_lo, m0_hi = [100 * x for x in m0["accuracy_bounds_without_imputation"]]
m0_mid = (m0_lo + m0_hi) / 2
cot_bounds = [
    [100 * x for x in summary["accuracy_bounds_without_imputation"]] for summary in cot_summaries
]
cot_acc = [(lo + hi) / 2 for lo, hi in cot_bounds]
coconut_acc = [
    100 * trajectory["stage1"]["k2"]["accuracy"],
    100 * trajectory["stage2"]["k4"]["accuracy"],
    100 * trajectory["stage3"]["k6"]["accuracy"],
]

x = [0, 1, 2, 3]

fig, ax = plt.subplots(figsize=(7.4, 3.55))
(cot_line,) = ax.plot(
    x,
    [m0_mid, *cot_acc],
    color=BLUE,
    marker="o",
    markersize=6,
    linewidth=2.0,
    label="Explicit CoT",
)
(coconut_line,) = ax.plot(
    x,
    [m0_mid, *coconut_acc],
    color=ORANGE,
    marker="o",
    markersize=6,
    linewidth=2.0,
    label="Coconut (trained K)",
)
m0_marker = ax.scatter(
    [0],
    [m0_mid],
    marker="D",
    s=48,
    color=GREY,
    edgecolor="white",
    linewidth=0.7,
    zorder=5,
    label="M0 (shared start)",
)
ax.errorbar(
    0,
    m0_mid,
    yerr=[[m0_mid - m0_lo], [m0_hi - m0_mid]],
    fmt="none",
    color="#333333",
    capsize=4,
    linewidth=1.2,
)
ax.set_ylabel("GSM8K-200 accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xlim(-0.20, 3.20)
ax.set_xticks(x)
ax.set_xticklabels(["M0", "u1", "u2", "u3"])
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
fig.suptitle("Capability through training", x=0.105, y=0.985, ha="left", fontsize=10.5)
fig.legend(
    handles=[m0_marker, cot_line, coconut_line],
    labels=["M0 (shared start)", "Explicit CoT", "Coconut (trained K)"],
    frameon=False,
    ncol=3,
    loc="upper left",
    bbox_to_anchor=(0.095, 0.905),
)

ax.text(
    0,
    m0_mid + 3.0,
    f"{m0_lo:.1f}–{m0_hi:.1f}",
    ha="center",
    va="bottom",
    color=GREY,
    fontsize=9,
)
for values, color, bounds, offset in (
    (cot_acc, BLUE, cot_bounds, 2.3),
    (coconut_acc, ORANGE, None, -3.0),
):
    for index, value in enumerate(values):
        label = f"{value:.1f}"
        if bounds is not None and bounds[index][0] != bounds[index][1]:
            label = f"{bounds[index][0]:.1f}–{bounds[index][1]:.1f}"
        ax.text(
            index + 1,
            value + offset,
            label,
            ha="center",
            va="bottom" if offset > 0 else "top",
            color=color,
            fontsize=9,
        )

for stage, value, (lo, hi) in zip((1, 2, 3), cot_acc, cot_bounds):
    if lo != hi:
        ax.errorbar(
            stage,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt="none",
            color="#333333",
            capsize=3,
            linewidth=1.0,
        )

fig.subplots_adjust(left=0.105, right=0.98, bottom=0.15, top=0.76)
fig.savefig(ROOT / "writeup/figures/fig1_capability.png", dpi=220)
plt.close(fig)
print("wrote fig1_capability.png")

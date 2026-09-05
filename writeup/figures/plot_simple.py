"""Summary figures for the application write-up. Reads committed artifacts only."""

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
LIGHT_GREY = "#D9D9D9"

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


# Figure 1: capability
m0 = json.load(open(RESULTS / "native_gsm8k_controls/m0/summary.json"))
cot = json.load(open(RESULTS / "native_gsm8k_controls/cot_u3/summary.json"))
trajectory = json.load(
    open(RESULTS / "fallback_4b_skip0/k_trajectory_consolidated.json")
)["gsm8k_200"]

m0_lo, m0_hi = [100 * x for x in m0["accuracy_bounds_without_imputation"]]
m0_mid = (m0_lo + m0_hi) / 2
cot_acc = 100 * cot["observed_accuracy"]
stage_acc = [
    100 * trajectory["stage1"]["k2"]["accuracy"],
    100 * trajectory["stage2"]["k4"]["accuracy"],
    100 * trajectory["stage3"]["k6"]["accuracy"],
]

labels = ["M0", "CoT\nu3", "Coconut\nu1, K=2", "Coconut\nu2, K=4", "Coconut\nu3, K=6"]
values = [m0_mid, cot_acc, *stage_acc]
colors = [GREY, BLUE, ORANGE, ORANGE, ORANGE]

fig, ax = plt.subplots(figsize=(7.4, 3.9))
bars = ax.bar(range(len(values)), values, color=colors, width=0.64)
ax.errorbar(
    0,
    m0_mid,
    yerr=[[m0_mid - m0_lo], [m0_hi - m0_mid]],
    fmt="none",
    color="#333333",
    capsize=4,
    linewidth=1.2,
)
ax.set_title("GSM8K-200 accuracy", loc="left")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
for i, (bar, value) in enumerate(zip(bars, values)):
    label = f"{m0_lo:.1f}-{m0_hi:.1f}" if i == 0 else f"{value:.1f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2.0,
        label,
        ha="center",
        va="bottom",
        color=colors[i],
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig1_capability.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# Figure 2: registered endpoint safety contrast
comparison = json.load(
    open(RESULTS / "official_safety/scores/paired_comparison.json")
)
delta = comparison["mean_delta"]
ci_lo, ci_hi = comparison["paired_prompt_bootstrap_95_ci"]

fig, ax = plt.subplots(figsize=(7.4, 2.5))
ax.axvspan(-0.02, 0.02, color=LIGHT_GREY, alpha=0.55)
ax.axvline(0, color=GREY, linestyle=":", linewidth=1.1)
ax.errorbar(
    delta,
    0,
    xerr=[[delta - ci_lo], [ci_hi - delta]],
    fmt="o",
    color=ORANGE,
    capsize=5,
    linewidth=2,
    markersize=7,
)
ax.text(
    delta,
    0.12,
    f"{delta:+.3f}  (95% CI {ci_lo:+.3f} to {ci_hi:+.3f})",
    ha="center",
    va="bottom",
    color="#333333",
    fontsize=9,
)
ax.set_title("Endpoint safety difference: Coconut minus CoT", loc="left")
ax.set_xlabel("Difference in mean StrongREJECT score")
ax.set_xlim(-0.055, 0.07)
ax.set_ylim(-0.32, 0.34)
ax.set_yticks([])
ax.grid(axis="x", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig2_endpoint_safety.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("wrote fig1_capability.png and fig2_endpoint_safety.png")

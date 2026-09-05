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


def label_point(ax, x, y, label, *, color="#333333", dx=0, dy=5, ha="center"):
    ax.annotate(
        label,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        color=color,
        fontsize=9,
    )


# Figure 1: capability
m0 = json.load(open(RESULTS / "native_gsm8k_controls/m0/summary.json"))
cot = json.load(open(RESULTS / "native_gsm8k_controls/cot_u3/summary.json"))
trajectory = json.load(
    open(RESULTS / "fallback_4b_skip0/k_trajectory_consolidated.json")
)["gsm8k_200"]

m0_lo, m0_hi = [100 * x for x in m0["accuracy_bounds_without_imputation"]]
cot_acc = 100 * cot["observed_accuracy"]
k6 = 100 * trajectory["stage3"]["k6"]["accuracy"]
k0_lo, k0_hi = [
    100 * x
    for x in trajectory["stage3"]["k0"]["accuracy_bounds_without_imputation"]
]
stage_acc = [
    100 * trajectory["stage1"]["k2"]["accuracy"],
    100 * trajectory["stage2"]["k4"]["accuracy"],
    k6,
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
for ax in (ax1, ax2):
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)

ax1.set_title("A. Endpoint accuracy", loc="left")
ax1.set_ylabel("GSM8K accuracy (%)")
ax1.set_xticks(range(4))
ax1.set_xticklabels(["M0", "CoT\nu3", "Coconut\nu3, K=6", "Coconut\nu3, K=0"])
ax1.plot([0, 0], [m0_lo, m0_hi], color=GREY, linewidth=8, solid_capstyle="butt")
ax1.plot(1, cot_acc, "o", color=BLUE, markersize=7)
ax1.plot(2, k6, "o", color=ORANGE, markersize=7)
ax1.plot(
    [3, 3],
    [k0_lo, k0_hi],
    color=ORANGE,
    alpha=0.45,
    linewidth=8,
    solid_capstyle="butt",
)
label_point(ax1, 0, m0_hi, f"{m0_lo:.1f}-{m0_hi:.1f}", color=GREY)
label_point(ax1, 1, cot_acc, f"{cot_acc:.1f}", color=BLUE)
label_point(ax1, 2, k6, f"{k6:.1f}", color=ORANGE)
label_point(ax1, 3, k0_hi, f"{k0_lo:.1f}-{k0_hi:.1f}", color=ORANGE)
ax1.text(3, k0_lo - 8, "20/200 did not terminate", ha="center", color="#555555", fontsize=8)

ax2.set_title("B. Coconut accuracy by training stage", loc="left")
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(["u1\nK=2", "u2\nK=4", "u3\nK=6"])
ax2.axhspan(m0_lo, m0_hi, color=LIGHT_GREY, alpha=0.55, linewidth=0)
ax2.axhline(cot_acc, color=BLUE, linestyle="--", linewidth=1.2)
ax2.plot([1, 2, 3], stage_acc, "o-", color=ORANGE, linewidth=1.8, markersize=6)
for x, y in zip([1, 2, 3], stage_acc):
    label_point(ax2, x, y, f"{y:.1f}", color=ORANGE)
ax2.text(1.03, m0_lo - 6, "M0 range", color=GREY, fontsize=8.5)
ax2.text(1.03, cot_acc + 2, "CoT u3", color=BLUE, fontsize=8.5)

fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig1_capability.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# Figure 2: safety trajectory
rows = {
    row["condition"]: row
    for row in json.load(open(RESULTS / "dense_safety/trajectory/trajectory.json"))["rows"]
}
seed43 = json.load(
    open(RESULTS / "matched_4b_cot_seed43/cot_u1_seed_comparison.json")
)["means"]["seed43"]

m0_score = rows["m0"]["mean_score"]
cot_scores = [
    m0_score,
    rows["cot_u1"]["mean_score"],
    rows["cot_u2"]["mean_score"],
    rows["cot_u3"]["mean_score"],
]
coconut_bounds = [
    (rows["coco_u1_k2"]["lower_bound"], rows["coco_u1_k2"]["upper_bound"]),
    (rows["coco_u2_k4"]["lower_bound"], rows["coco_u2_k4"]["upper_bound"]),
]
coconut_endpoint = rows["coco_u3_k6"]["mean_score"]

fig, ax = plt.subplots(figsize=(7.6, 4.1))
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
ax.axhline(m0_score, color=GREY, linestyle=":", linewidth=1.1, label="M0")
ax.plot([0, 1, 2, 3], cot_scores, "o-", color=BLUE, linewidth=1.8, markersize=6, label="CoT")
ax.plot(1, seed43, "o", markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.5, markersize=6, label="CoT u1, seed 43")

for x, (lo, hi) in zip([1, 2], coconut_bounds):
    ax.plot([x + 0.10, x + 0.10], [lo, hi], color=ORANGE, alpha=0.55, linewidth=9, solid_capstyle="butt")
for x, (_, hi), n in zip([1, 2], coconut_bounds, [2, 3]):
    ax.text(x + 0.15, hi + 0.002, f"{n}/60 unfinished", color=ORANGE, fontsize=7.8)
ax.plot(3, coconut_endpoint, "s", color=ORANGE, markersize=6, label="Coconut")

for x, y in zip([1, 2, 3], cot_scores[1:]):
    label_point(ax, x, y, f"{y:.3f}", color=BLUE, dx=-4, dy=5)
label_point(ax, 1, seed43, f"{seed43:.3f}", color=BLUE, dx=5, dy=-15, ha="left")
label_point(ax, 3, coconut_endpoint, f"{coconut_endpoint:.3f}", color=ORANGE, dx=5, dy=3, ha="left")

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["M0", "u1", "u2", "u3"])
ax.set_xlim(-0.35, 3.55)
ax.set_ylim(0.04, 0.15)
ax.set_ylabel("Mean StrongREJECT score")
ax.legend(frameon=False, ncol=4, fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig2_trajectory.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# Figure 3: descriptive endpoint weight changes
weight_diff = json.load(
    open(RESULTS / "posthoc_layerwise_weight_diff/layerwise_weight_updates.json")
)
layers = [row for row in weight_diff["layers"] if row["group"].startswith("layer_")]
layer_ids = [int(row["group"].split("_")[1]) for row in layers]
cot_relative = [100 * row["cot_relative_l2"] for row in layers]
coconut_relative = [100 * row["coconut_relative_l2"] for row in layers]
alignment = [row["update_cosine"] for row in layers]

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.45, 1]}
)
for ax in (ax1, ax2):
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)

ax1.set_title("A. Relative update size", loc="left")
ax1.plot(layer_ids, cot_relative, color=BLUE, linewidth=1.8, label="Explicit CoT")
ax1.plot(layer_ids, coconut_relative, color=ORANGE, linewidth=1.8, label="Coconut")
ax1.set_xlabel("Transformer layer")
ax1.set_ylabel("Update L2 / M0 weight L2 (%)")
ax1.legend(frameon=False)

ax2.set_title("B. Alignment between updates", loc="left")
ax2.plot(layer_ids, alignment, color=GREY, linewidth=1.8)
ax2.axhline(0, color=LIGHT_GREY, linewidth=1)
ax2.set_xlabel("Transformer layer")
ax2.set_ylabel("Cosine similarity")

fig.tight_layout()
fig.savefig(ROOT / "writeup/figures/fig3_weight_updates.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("wrote fig1_capability.png, fig2_trajectory.png, and fig3_weight_updates.png")

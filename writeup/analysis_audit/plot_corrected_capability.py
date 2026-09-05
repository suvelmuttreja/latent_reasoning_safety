"""Plot corrected missing-outcome bounds from the independent audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
data = json.loads((root / "recomputed_checks.json").read_text())["capability"]
fig, ax = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
stages = [1, 2, 3]
ax.plot(
    stages,
    [100 * data[f"coco_u{s}_k{2 * s}"]["all_row_cutoff_parser_accuracy"] for s in stages],
    "o-",
    color="#b74c2a",
    label="Coconut at trained K=2/4/6",
)
for s in stages:
    result = data[f"coco_u{s}_k0"]
    lo, hi = [100 * v for v in result["unknown_unfinished_outcome_bounds"]]
    ax.plot([s, s], [lo, hi], color="#276e91", linewidth=3)
    ax.plot([s - 0.025, s + 0.025], [lo, lo], color="#276e91")
    ax.plot([s - 0.025, s + 0.025], [hi, hi], color="#276e91")
    ax.plot(
        s,
        100 * result["all_row_cutoff_parser_accuracy"],
        "o",
        color="#276e91",
        label="Coconut K=0: cutoff accuracy + outcome bounds" if s == 1 else None,
    )
    ax.annotate(
        f"{lo:g}–{hi:g}%",
        (s, hi),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color="#276e91",
        fontsize=10,
    )
ax.scatter([3], [92], marker="D", color="#326b44", label="Explicit CoT u3 (92%)", zorder=4)
ax.axhspan(87.5, 89.5, color="#999999", alpha=0.2, label="M0 outcome bounds (87.5–89.5%)")
ax.set(
    xticks=stages,
    xticklabels=["u1", "u2", "u3"],
    ylim=(20, 102),
    ylabel="GSM8K-200 accuracy (%)",
    title="The K=0 accuracy advantage survives corrected bounds",
)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", fontsize=9, frameon=False)
fig.get_layout_engine().set(rect=(0, 0.085, 1, 0.915))
fig.text(
    0.5,
    0.025,
    "Ranges treat unfinished outcomes as unknown; they are not confidence intervals.\n"
    "K=0 is an inference ablation. Earlier CoT capability controls are pending.",
    ha="center",
    fontsize=9,
    color="#444444",
)
fig.savefig(root / "corrected_capability.png", dpi=180)

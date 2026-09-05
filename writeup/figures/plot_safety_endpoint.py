"""Plot absolute endpoint StrongREJECT means from committed score files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "artifacts/discovery/results/official_safety/scores"

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

conditions = [
    ("M0", "m0.json", GREY),
    ("Explicit CoT\nu3", "cot_u3.json", BLUE),
    ("Coconut\nu3, K=6", "coco_u3_k6.json", ORANGE),
]

rng = np.random.default_rng(42)
means = []
intervals = []
colors = []
labels = []

for label, filename, color in conditions:
    payload = json.loads((SCORES / filename).read_text())
    values = np.asarray([row["score"] for row in payload["records"]], dtype=float)
    boot_indices = rng.integers(0, len(values), size=(10_000, len(values)))
    bootstrap_means = values[boot_indices].mean(axis=1)
    lo, hi = np.quantile(bootstrap_means, [0.025, 0.975])
    means.append(float(values.mean()))
    intervals.append((float(lo), float(hi)))
    colors.append(color)
    labels.append(label)

x = np.arange(len(means))
fig, ax = plt.subplots(figsize=(6.7, 2.8))
bars = ax.bar(x, means, width=0.5, color=colors)
ax.errorbar(
    x,
    means,
    yerr=[
        [mean - lo for mean, (lo, _) in zip(means, intervals)],
        [hi - mean for mean, (_, hi) in zip(means, intervals)],
    ],
    fmt="none",
    ecolor="#333333",
    capsize=4,
    linewidth=1.2,
)
ax.set_title("Endpoint harmfulness scores", loc="left", pad=12)
ax.set_ylabel("Mean StrongREJECT score\n(higher = more harmful)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.16)
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)

for bar, mean, color, (_, hi) in zip(bars, means, colors, intervals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        hi + 0.005,
        f"{mean:.3f}",
        ha="center",
        va="bottom",
        color=color,
        fontsize=9,
    )

fig.subplots_adjust(left=0.15, right=0.98, bottom=0.23, top=0.86)
fig.savefig(ROOT / "writeup/figures/fig2_endpoint_safety.png", dpi=220)
plt.close(fig)
print("wrote fig2_endpoint_safety.png")

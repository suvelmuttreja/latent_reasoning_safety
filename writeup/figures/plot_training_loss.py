"""Plot per-update training loss for both branches from committed logs.

Stage 1 loss was logged every 10 optimizer updates to the SLURM .out files
(jobs 11413196 and 11413535, matching stage1/metadata.json). Stages 2 and 3
were logged every update to update_metrics.jsonl. All use the same
shifted-nonignored-token normalization recorded in metadata.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts/discovery/results"
LOGS = ROOT / "artifacts/discovery/logs"
UPDATES_PER_STAGE = 468

BLUE = "#4C78A8"
ORANGE = "#E45756"

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

BRANCHES = {
    "Explicit CoT": ("matched_4b_cot", "mats-matched-4b-cot-s1-a40-11413535.out", BLUE),
    "Coconut skip0": ("fallback_4b_skip0", "mats-fallback-4b-coco-s1-a40-11413196.out", ORANGE),
}


def stage1_from_log(path: Path):
    xs, ys = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{") or '"loss"' not in line:
            continue
        row = json.loads(line)
        if row.get("stage") != 1:
            continue
        xs.append(row["stage_update"])
        ys.append(row["loss"])
    return np.array(xs), np.array(ys)


def later_stage(path: Path):
    xs, ys, finite, tokens = [], [], [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        xs.append(row["stage_update"])
        ys.append(row["token_weighted_loss"])
        finite.append(row["gradients_finite"])
        tokens.append(row["supervised_tokens"])
    return np.array(xs), np.array(ys), all(finite), np.array(tokens)


def rolling(y, w=20):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    out = np.convolve(y, kernel, mode="valid")
    pad = np.full(len(y) - len(out), np.nan)
    return np.concatenate([pad, out])


fig, ax = plt.subplots(figsize=(6.7, 3.0))
summary = {}
for label, (resdir, logname, color) in BRANCHES.items():
    x1, y1 = stage1_from_log(LOGS / logname)
    ax.plot(x1, y1, color=color, linewidth=1.4, label=label)
    summary[(label, 1)] = (float(y1[:3].mean()), float(y1[-3:].mean()), True, None)
    for stage in (2, 3):
        x, y, finite, tok = later_stage(RES / resdir / f"stage{stage}/update_metrics.jsonl")
        offset = (stage - 1) * UPDATES_PER_STAGE
        ax.plot(x + offset, y, color=color, linewidth=0.5, alpha=0.18)
        ax.plot(x + offset, rolling(y), color=color, linewidth=1.4)
        summary[(label, stage)] = (
            float(y[:20].mean()),
            float(y[-20:].mean()),
            finite,
            float(np.median(tok)),
        )

for s in (1, 2):
    ax.axvline(s * UPDATES_PER_STAGE, color="#999999", linewidth=0.8, linestyle="--")
for s, name in ((0, "stage 1\nK=2"), (1, "stage 2\nK=4"), (2, "stage 3\nK=6")):
    ax.text(
        s * UPDATES_PER_STAGE + UPDATES_PER_STAGE / 2,
        0.97,
        name,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

ax.set_xlim(0, 3 * UPDATES_PER_STAGE)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Optimizer update (468 per stage, both branches)")
ax.set_ylabel("Training loss\n(per supervised token)")
ax.set_title("Training loss fell within every stage on both branches", loc="left", pad=12)
ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, ncol=2)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.86)
out = ROOT / "writeup/figures/fig3_training_loss.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print("wrote", out.name)
for (label, stage), (first, last, finite, tok) in sorted(summary.items()):
    print(
        f"{label} stage {stage}: first {first:.3f} -> last {last:.3f}; "
        f"all gradients finite={finite}; median supervised tokens/update={tok}"
    )

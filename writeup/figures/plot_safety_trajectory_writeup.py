"""Write-up version of the dense safety trajectory plot.

Same data and same reading as scripts/plot_dense_safety_trajectory.py, plus a
Coconut trajectory drawn as a band: its edges connect M0 -> u1 lower/upper
bound -> u2 lower/upper bound -> u3 point. No midpoint line is drawn, because
the u1/u2 cells have bounds rather than point estimates.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "artifacts/discovery/results/dense_safety/trajectory/trajectory.json"
U1 = ROOT / "artifacts/discovery/results/dense_safety/trajectory/cot_minus_coconut_u1_bounded_bootstrap.json"
SEED = ROOT / "artifacts/discovery/results/matched_4b_cot_seed43/cot_u1_seed_comparison.json"

BLUE = "#2563eb"
RED = "#dc2626"

payload = json.loads(TRAJ.read_text())
assert payload["status"] == "complete"
rows = {row["condition"]: row for row in payload["rows"]}
u1 = json.loads(U1.read_text())
assert u1["status"] == "complete" and u1["effect"] == "cot_u1_minus_coco_u1_k2"
seed = json.loads(SEED.read_text())
assert seed["status"] == "complete"

fig, ax = plt.subplots(figsize=(10.4, 5.4))

# Explicit CoT line
cot_y = [rows[n]["mean_score"] for n in ("m0", "cot_u1", "cot_u2", "cot_u3")]
ax.plot([0, 1, 2, 3], cot_y, color=BLUE, marker="o", linewidth=2.4, markersize=7,
        label="Explicit CoT seed 42", zorder=3)
ax.scatter([1], [seed["means"]["seed43"]], facecolors="white", edgecolors=BLUE, marker="o",
           linewidths=2.0, s=70, label="Explicit CoT seed 43 (u1 only)", zorder=5)

# Coconut trajectory band: M0 point -> u1 bounds -> u2 bounds -> u3 point
m0 = rows["m0"]["mean_score"]
u3 = rows["coco_u3_k6"]["mean_score"]
lo = [m0, rows["coco_u1_k2"]["lower_bound"], rows["coco_u2_k4"]["lower_bound"], u3]
hi = [m0, rows["coco_u1_k2"]["upper_bound"], rows["coco_u2_k4"]["upper_bound"], u3]
xs = [0, 1, 2, 3]
ax.fill_between(xs, lo, hi, color=RED, alpha=0.12, linewidth=0, zorder=1,
                label="Coconut trajectory (bound band)")
ax.plot(xs, lo, color=RED, linewidth=1.4, linestyle="--", zorder=2)
ax.plot(xs, hi, color=RED, linewidth=1.4, linestyle="--", zorder=2)

ax.scatter([0, 3], [m0, u3], color=RED, marker="D", s=58, label="Coconut (scoreable cells)", zorder=4)
ax.scatter([0], [m0], color="#111827", marker="o", s=62, label="Shared M0", zorder=5)

for x, name, ceiling in ((1, "coco_u1_k2", "16,000"), (2, "coco_u2_k4", "5,120")):
    row = rows[name]
    lower, upper = row["lower_bound"], row["upper_bound"]
    mid = (lower + upper) / 2
    ax.errorbar([x], [mid], yerr=[[mid - lower], [upper - mid]], color=RED, marker="_",
                markersize=12, capsize=6, linewidth=2.4,
                label="Coconut worst-case bound" if x == 1 else None, zorder=4)
    ax.text(x, upper + 0.004, f"{row['nonterminating_rows']}/60 nontermination\n{ceiling}-token ceiling",
            ha="center", va="bottom", fontsize=8, color="#991b1b")

ax.axhline(m0, color="#6b7280", linewidth=1, linestyle=":")
ax.set_xticks([0, 1, 2, 3], ["M0", "Stage 1", "Stage 2", "Stage 3"])
top = max(rows[n]["upper_bound"] for n in ("coco_u1_k2", "coco_u2_k4"))
ax.set_ylim(0, max(0.16, top + 0.035))
ax.set_ylabel("Mean StrongREJECT harmfulness")
ax.set_title("M0-anchored matched safety trajectory (60 frozen prompts)")
ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
fig.subplots_adjust(left=0.09, right=0.70, bottom=0.18, top=0.90)

s42 = seed["bounded_substrate_contrasts"]["seed42"]
s43 = seed["bounded_substrate_contrasts"]["seed43"]
assert s42["identified_set"] == u1["identified_set"]
note = (
    "Stage-1 CoT − Coconut\n"
    f"seed 42 set: [{s42['identified_set'][0]:+.3f}, {s42['identified_set'][1]:+.3f}]\n"
    f"95% region: [{s42['conservative_identified_set_95_confidence_region'][0]:+.3f}, "
    f"{s42['conservative_identified_set_95_confidence_region'][1]:+.3f}]\n\n"
    f"seed 43 set: [{s43['identified_set'][0]:+.3f}, {s43['identified_set'][1]:+.3f}]\n"
    f"95% region: [{s43['conservative_identified_set_95_confidence_region'][0]:+.3f}, "
    f"{s43['conservative_identified_set_95_confidence_region'][1]:+.3f}]\n\n"
    "Paired-prompt uncertainty only;\none shared Coconut seed"
)
fig.text(0.73, 0.83, note, ha="left", va="top", fontsize=8.5, linespacing=1.25, color="#111827",
         bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f9fafb", "edgecolor": "#d1d5db", "linewidth": 0.9})
fig.text(0.5, 0.055, "Coconut u1/u2 bars are bounds, not point estimates; the shaded band connects those bounds.",
         ha="center", fontsize=8, color="#4b5563")
fig.text(0.5, 0.025, "Nonterminations use the theoretical StrongREJECT extrema: 0.0 for the lower bound and 1.0 for the upper bound.",
         ha="center", fontsize=8, color="#4b5563")

out = ROOT / "writeup/figures/fig4_safety_trajectory.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print("wrote", out.name)

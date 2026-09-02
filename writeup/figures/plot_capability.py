"""Figure 1: matched-branch GSM8K-200 capability. Plot-only; reads committed artifacts."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "artifacts/discovery/results"
m0 = json.load(open(R / "native_gsm8k_controls/m0/summary.json"))
cot = json.load(open(R / "native_gsm8k_controls/cot_u3/summary.json"))
traj = json.load(open(R / "fallback_4b_skip0/k_trajectory_consolidated.json"))["gsm8k_200"]

m0_lo, m0_hi = [100 * x for x in m0["accuracy_bounds_without_imputation"]]
cot_acc = 100 * cot["observed_accuracy"]
s1, s2, s3 = traj["stage1"], traj["stage2"], traj["stage3"]
k0_lo, k0_hi = [100 * x for x in s3["k0"]["accuracy_bounds_without_imputation"]]
k6 = 100 * s3["k6"]["accuracy"]
n_nonterm = s3["k0"]["length_stops_at_1024"]
n_cycles = s3["k0"]["exact_token_cycles_among_1024_length_stops"]

BLUE, RED, INK, MUTED, GRID = "#2563eb", "#dc2626", "#111827", "#4b5563", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                           gridspec_kw={"width_ratios": [1.05, 1]})
for ax in (a, b):
    ax.set_ylim(0, 100); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
a.set_ylabel("GSM8K-200 accuracy (%)")

# Panel A: endpoint matched comparison
xs = [0, 1, 2, 3]
a.set_xticks(xs); a.set_xticklabels(["M0\n(base)", "Explicit CoT\nu3", "Our skip0 u3\nK=6 (trained)", "Our skip0 u3\nK=0 (ablated)"])
a.plot([0, 0], [m0_lo, m0_hi], color=INK, lw=6, solid_capstyle="butt")
a.text(0.13, (m0_lo + m0_hi) / 2, f"{m0_lo:.1f}–{m0_hi:.1f}", va="center", fontsize=9)
a.plot(1, cot_acc, "o", color=BLUE, ms=9); a.text(1.13, cot_acc, f"{cot_acc:.1f}", va="center", fontsize=9)
a.plot(2, k6, "D", color=RED, ms=8); a.text(2.13, k6, f"{k6:.1f}", va="center", fontsize=9)
a.plot([3, 3], [k0_lo, k0_hi], color=RED, lw=6, alpha=0.45, solid_capstyle="butt")
a.text(3.13, (k0_lo + k0_hi) / 2, f"{k0_lo:.1f}–{k0_hi:.1f}", va="center", fontsize=9)
a.annotate(f"{n_nonterm}/200 never terminate\n({n_cycles} exact token cycles)", xy=(3, k0_lo), xytext=(2.35, 12),
           fontsize=8.5, color=RED, ha="left", arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
a.annotate("", xy=(2, k6 + 3), xytext=(2, cot_acc - 3), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
a.text(1.55, (k6 + cot_acc) / 2, f"−{cot_acc - k6:.0f} pts", ha="right", va="center", fontsize=9, color=MUTED)
a.set_xlim(-0.5, 3.9)
a.set_title("A. Same benign GSM8K post-training, matched", loc="left", fontsize=11)

# Panel B: Coconut curriculum trajectory
st = [1, 2, 3]
b.set_xticks(st); b.set_xticklabels(["u1\n(K=2)", "u2\n(K=4)", "u3\n(K=6)"]); b.set_xlim(0.6, 3.7)
b.axhspan(m0_lo, m0_hi, color=INK, alpha=0.12, lw=0); b.text(0.65, m0_lo - 4.5, "M0 range", fontsize=8.5, color=MUTED)
b.axhline(cot_acc, color=BLUE, lw=1.2, ls=":"); b.text(0.65, cot_acc + 2, "Explicit CoT u3", fontsize=8.5, color=BLUE)
kt = [100 * s1["k2"]["accuracy"], 100 * s2["k4"]["accuracy"], k6]
k0 = [100 * s1["k0"]["accuracy"], 100 * s2["k0"]["accuracy"]]
b.plot(st, kt, "-D", color=RED, lw=2, ms=7, label="trained K (latents on)")
b.plot(st[:2], k0, "--o", color=RED, lw=1.6, ms=7, mfc="white", alpha=0.7, label="K=0 (latents removed)")
b.plot([3, 3], [k0_lo, k0_hi], color=RED, lw=6, alpha=0.45, solid_capstyle="butt")
for x, y in zip(st, kt): b.text(x, y - 6, f"{y:.1f}", ha="center", fontsize=9, color=RED)
for x, y in zip(st[:2], k0): b.text(x, y + 3, f"{y:.1f}", ha="center", fontsize=9, color=RED, alpha=0.8)
b.text(3.08, (k0_lo + k0_hi) / 2, f"{k0_lo:.1f}–{k0_hi:.1f}\n({n_nonterm} nonterminating)", va="center", fontsize=8.5, color=RED, alpha=0.9)
b.legend(loc="lower left", frameon=False, fontsize=9)
b.set_title("B. Our skip0 branch, stage by stage", loc="left", fontsize=11)

fig.text(0.01, 0.005, "Ranges are no-imputation bounds: nonterminating rows counted as all wrong / all right. "
         "Stages 1–2 at a 512-token cap; stage 3 regenerated at 1,024. Greedy decoding, 200 fixed test items.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = ROOT / "writeup/figures/fig1_capability"
fig.savefig(f"{out}.png", dpi=200)
print("wrote", f"{out}.png")

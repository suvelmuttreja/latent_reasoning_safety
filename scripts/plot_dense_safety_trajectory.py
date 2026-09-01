#!/usr/bin/env python3
"""Render point estimates and partial-identification safety bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        default="artifacts/discovery/results/dense_safety/trajectory/trajectory.json",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/discovery/results/dense_safety/trajectory"
    )
    parser.add_argument(
        "--u1-comparison",
        default=(
            "artifacts/discovery/results/dense_safety/trajectory/"
            "cot_minus_coconut_u1_bounded_bootstrap.json"
        ),
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.trajectory).read_text())
    if payload["status"] != "complete":
        raise ValueError("refusing to render an incomplete trajectory")
    rows = {row["condition"]: row for row in payload["rows"]}
    u1 = json.loads(Path(args.u1_comparison).read_text())
    if u1["status"] != "complete":
        raise ValueError("refusing to render an incomplete u1 comparison")
    if u1["effect"] != "cot_u1_minus_coco_u1_k2":
        raise ValueError("unexpected u1 comparison orientation")

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    explicit_names = ["m0", "cot_u1", "cot_u2", "cot_u3"]
    explicit_y = [rows[name]["mean_score"] for name in explicit_names]
    ax.plot(
        [0, 1, 2, 3],
        explicit_y,
        color="#2563eb",
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="Explicit CoT",
        zorder=3,
    )
    ax.scatter(
        [0, 3],
        [rows["m0"]["mean_score"], rows["coco_u3_k6"]["mean_score"]],
        color="#dc2626",
        marker="D",
        s=58,
        label="Coconut (scoreable cells)",
        zorder=4,
    )
    ax.scatter(
        [0],
        [rows["m0"]["mean_score"]],
        color="#111827",
        marker="o",
        s=62,
        label="Shared M0",
        zorder=5,
    )

    for x, name, ceiling in (
        (1, "coco_u1_k2", "16,000"),
        (2, "coco_u2_k4", "5,120"),
    ):
        row = rows[name]
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        if lower is None or upper is None:
            raise ValueError(f"missing partial-identification bounds for {name}")
        midpoint = (lower + upper) / 2
        ax.errorbar(
            [x],
            [midpoint],
            yerr=[[midpoint - lower], [upper - midpoint]],
            color="#dc2626",
            marker="_",
            markersize=12,
            capsize=6,
            linewidth=2.4,
            label="Coconut worst-case bound" if x == 1 else None,
            zorder=4,
        )
        ax.text(
            x,
            upper + 0.004,
            f"{row['nonterminating_rows']}/60 nontermination\n{ceiling}-token ceiling",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#991b1b",
        )

    ax.axhline(rows["m0"]["mean_score"], color="#6b7280", linewidth=1, linestyle=":")
    ax.set_xticks([0, 1, 2, 3], ["M0", "Stage 1", "Stage 2", "Stage 3"])
    highest_bound = max(rows[name]["upper_bound"] for name in ("coco_u1_k2", "coco_u2_k4"))
    ax.set_ylim(0, max(0.16, highest_bound + 0.035))
    ax.set_ylabel("Mean StrongREJECT harmfulness")
    ax.set_title("M0-anchored matched safety trajectory (60 frozen prompts)")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.70, bottom=0.18, top=0.90)

    identified = u1["identified_set"]
    lower_ci = u1["lower_endpoint_bootstrap_95_ci"]
    upper_ci = u1["upper_endpoint_bootstrap_95_ci"]
    confidence_region = u1["conservative_identified_set_95_confidence_region"]
    annotation = (
        "Stage-1 CoT − Coconut\n"
        f"identified set: [{identified[0]:+.3f}, {identified[1]:+.3f}]\n\n"
        "Paired-prompt bootstrap (95%)\n"
        f"lower endpoint: [{lower_ci[0]:+.3f}, {lower_ci[1]:+.3f}]\n"
        f"upper endpoint: [{upper_ci[0]:+.3f}, {upper_ci[1]:+.3f}]\n"
        f"conservative region: [{confidence_region[0]:+.3f}, "
        f"{confidence_region[1]:+.3f}]\n\n"
        "Prompt uncertainty only;\n"
        "single training seed"
    )
    fig.text(
        0.73,
        0.83,
        annotation,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.25,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f9fafb",
            "edgecolor": "#d1d5db",
            "linewidth": 0.9,
        },
    )
    fig.text(
        0.5,
        0.055,
        "Coconut u1/u2 bars are bounds, not point estimates. Nonterminations use the theoretical StrongREJECT",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.text(
        0.5,
        0.025,
        "extrema: 0.0 for the lower bound and 1.0 for the upper bound—not the observed maximum.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "trajectory.png", dpi=220)
    fig.savefig(output_dir / "trajectory.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

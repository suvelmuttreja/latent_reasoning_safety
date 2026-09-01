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
    args = parser.parse_args()

    payload = json.loads(Path(args.trajectory).read_text())
    if payload["status"] != "complete":
        raise ValueError("refusing to render an incomplete trajectory")
    rows = {row["condition"]: row for row in payload["rows"]}

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
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
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(
        0.5,
        0.025,
        "Coconut u1/u2 bars are extreme-outcome bounds, not point estimates: every nontermination is set to 0 or 1.",
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

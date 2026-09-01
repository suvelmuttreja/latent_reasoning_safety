#!/usr/bin/env python3
"""Plot the post-hoc descriptive endpoint weight-update comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "artifacts/discovery/results/posthoc_layerwise_weight_diff/"
            "layerwise_weight_updates.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/discovery/results/posthoc_layerwise_weight_diff",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    if payload["status"] != "complete":
        raise ValueError("refusing to plot an incomplete weight comparison")
    layers = [row for row in payload["layers"] if row["group"].startswith("layer_")]
    layer_ids = [int(row["group"].split("_")[1]) for row in layers]
    cot_relative = [100 * row["cot_relative_l2"] for row in layers]
    coconut_relative = [100 * row["coconut_relative_l2"] for row in layers]
    alignment = [row["update_cosine"] for row in layers]

    fig, (ax_norm, ax_alignment) = plt.subplots(
        1, 2, figsize=(11.2, 5.2), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    ax_norm.plot(
        layer_ids,
        cot_relative,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Explicit CoT",
    )
    ax_norm.plot(
        layer_ids,
        coconut_relative,
        color="#dc2626",
        linewidth=2.2,
        marker="D",
        markersize=3.2,
        label="Coconut",
    )
    ax_norm.set_xlabel("Transformer layer")
    ax_norm.set_ylabel("Update L2 / M0 weight L2 (%)")
    ax_norm.set_title("A. Relative parameter movement")
    ax_norm.legend(frameon=False)
    ax_norm.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    ax_alignment.plot(
        layer_ids,
        alignment,
        color="#7c3aed",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
    )
    ax_alignment.axhline(0, color="#9ca3af", linewidth=1, linestyle=":")
    ax_alignment.set_xlabel("Transformer layer")
    ax_alignment.set_ylabel("Cosine(CoT update, Coconut update)")
    ax_alignment.set_title("B. Update-direction alignment")
    ax_alignment.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    embedding = next(row for row in payload["layers"] if row["group"] == "embeddings")
    overall = payload["overall"]
    note = (
        f"Overall relative update: CoT {100 * overall['cot_relative_l2']:.3f}%, "
        f"Coconut {100 * overall['coconut_relative_l2']:.3f}%\n"
        f"Shared embedding rows: CoT {100 * embedding['cot_relative_l2']:.3f}%, "
        f"Coconut {100 * embedding['coconut_relative_l2']:.3f}%; "
        f"cosine {embedding['update_cosine']:.3f}"
    )
    fig.suptitle("Post-hoc endpoint weight updates relative to exact M0", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.82, wspace=0.30)
    fig.text(0.5, 0.14, note, ha="center", fontsize=8.5, color="#374151")
    fig.text(
        0.5,
        0.045,
        "Descriptive only: parameter movement is not causal localization of capability loss. "
        "Vocabulary matrices use the shared token-ID prefix.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    for ax in (ax_norm, ax_alignment):
        ax.spines[["top", "right"]].set_visible(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "layerwise_weight_updates.png", dpi=220)
    fig.savefig(output_dir / "layerwise_weight_updates.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

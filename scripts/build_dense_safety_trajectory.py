#!/usr/bin/env python3
"""Build the M0-anchored matched safety trajectory table from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from mats_latent_safety.hashing import sha256_file


CONDITIONS = (
    ("m0", "base", 0, "artifacts/discovery/results/official_safety/scores/m0.json"),
    ("cot_u1", "explicit_cot", 1, "artifacts/discovery/results/dense_safety/scores/cot_u1.json"),
    ("cot_u2", "explicit_cot", 2, "artifacts/discovery/results/dense_safety/scores/cot_u2.json"),
    ("cot_u3", "explicit_cot", 3, "artifacts/discovery/results/official_safety/scores/cot_u3.json"),
    ("coco_u1_k2", "coconut_skip0", 1, None),
    ("coco_u2_k4", "coconut_skip0", 2, "artifacts/discovery/results/dense_safety/scores/coco_u2_k4.json"),
    ("coco_u3_k6", "coconut_skip0", 3, "artifacts/discovery/results/official_safety/scores/coco_u3_k6.json"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_safety_trajectory.yaml")
    parser.add_argument(
        "--output-dir", default="artifacts/discovery/results/dense_safety/trajectory"
    )
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    rows = []
    for condition, branch, stage, score_name in CONDITIONS:
        registered = config["conditions"][condition]
        score_path = Path(score_name) if score_name else None
        structural_missingness = registered["status"].startswith("scalar_unavailable")
        row = {
            "condition": condition,
            "branch": branch,
            "stage": stage,
            "status": registered["status"],
            "mean_score": None,
            "delta_from_m0": None,
            "score_sha256": None,
            "structural_missingness": structural_missingness,
        }
        if score_path is not None and score_path.exists():
            score = json.loads(score_path.read_text())
            if score.get("official_condition") != condition:
                raise ValueError(f"score condition mismatch for {condition}")
            observed_hash = sha256_file(score_path)
            registered_hash = registered.get("score_sha256")
            if registered_hash and observed_hash != registered_hash:
                raise ValueError(f"score hash mismatch for {condition}")
            row.update(
                mean_score=float(score["mean_score"]),
                score_sha256=observed_hash,
            )
        elif score_path is not None and not args.allow_pending and not structural_missingness:
            raise FileNotFoundError(f"missing score for {condition}: {score_path}")
        if condition == "coco_u1_k2":
            row.update(
                nontermination="3/60 at 16000 tokens",
                missingness_reason="strict <5% cap guard failed exactly at 5%",
            )
        elif condition == "coco_u2_k4":
            row.update(
                nontermination="3/60 at 5120 tokens",
                missingness_reason="strict <5% cap guard failed exactly at 5%",
            )
        rows.append(row)

    m0 = next(row["mean_score"] for row in rows if row["condition"] == "m0")
    if m0 is None:
        raise ValueError("M0 anchor score is required")
    for row in rows:
        if row["mean_score"] is not None:
            row["delta_from_m0"] = row["mean_score"] - m0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": (
            "complete"
            if all(row["mean_score"] is not None or row["structural_missingness"] for row in rows)
            else "pending_registered_conditions"
        ),
        "config_sha256": sha256_file(config_path),
        "m0_anchor": m0,
        "rows": rows,
        "interpretation_guard": (
            "Structural missingness is not zero; deltas are descriptive point estimates."
        ),
    }
    json_path = output_dir / "trajectory.json"
    csv_path = output_dir / "trajectory.csv"
    md_path = output_dir / "trajectory.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = (
        "condition",
        "branch",
        "stage",
        "status",
        "mean_score",
        "delta_from_m0",
        "structural_missingness",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Condition | Branch | Stage | Status | Mean | Delta from M0 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        mean = "—" if row["mean_score"] is None else f'{row["mean_score"]:.6f}'
        delta = "—" if row["delta_from_m0"] is None else f'{row["delta_from_m0"]:+.6f}'
        lines.append(
            f'| {row["condition"]} | {row["branch"]} | {row["stage"]} | '
            f'{row["status"]} | {mean} | {delta} |'
        )
    lines.extend(
        [
            "",
            "Structural missingness is shown as unavailable, never as zero.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows)}))


if __name__ == "__main__":
    main()

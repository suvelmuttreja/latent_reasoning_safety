#!/usr/bin/env python3
"""Merge the sealed blind human audit with continuous judge scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from mats_latent_safety.hashing import sha256_file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--human", required=True)
    parser.add_argument("--score", action="append", required=True, metavar="CONDITION=PATH")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    key_path = Path(args.key)
    human_path = Path(args.human)
    key_payload = json.loads(key_path.read_text())
    key_rows = {row["blind_id"]: row for row in key_payload["records"]}
    human_rows = read_jsonl(human_path)
    if set(key_rows) != {row["blind_id"] for row in human_rows}:
        raise ValueError("human packet and sealed condition key do not match")

    score_paths = {}
    score_rows = {}
    for item in args.score:
        condition, raw_path = item.split("=", 1)
        path = Path(raw_path)
        score_paths[condition] = path
        payload = json.loads(path.read_text())
        score_rows[condition] = {row["prompt_id"]: row for row in payload["records"]}

    merged = []
    for human in human_rows:
        key = key_rows[human["blind_id"]]
        condition = key["source_condition"]
        prompt_id = key["prompt_id"]
        score = score_rows[condition][prompt_id]
        merged.append(
            {
                "condition": condition,
                "prompt_id": prompt_id,
                "blind_id": human["blind_id"],
                "category": human["category"],
                "human_label": human["human_label"],
                "coherent": bool(human["coherent"]),
                "judge_score": float(score["score"]),
                "full_answer_tokens": score.get("full_answer_tokens"),
                "full_generated_tokens": score.get("full_generated_tokens"),
                "human_note": human.get("note", ""),
            }
        )
    merged.sort(key=lambda row: (row["condition"], row["prompt_id"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "judge_human_table.json"
    csv_path = output_dir / "judge_human_table.csv"
    md_path = output_dir / "judge_human_table.md"
    json_path.write_text(json.dumps(merged, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_coherence: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for row in merged:
        grouped[(row["condition"], row["human_label"])].append(row["judge_score"])
        grouped_coherence[(row["condition"], row["coherent"])].append(
            row["judge_score"]
        )
    lines = [
        "# Judge versus blind human audit",
        "",
        "The automatic judge remains continuous. No post-hoc threshold was chosen to turn it into a human-label classifier.",
        "",
        "| Condition | Prompt | Human label | Coherent | Judge score | Answer tokens | Generated tokens |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in merged:
        lines.append(
            f"| {row['condition']} | {row['prompt_id']} | {row['human_label']} | "
            f"{str(row['coherent']).lower()} | {row['judge_score']:.6f} | "
            f"{row['full_answer_tokens']} | {row['full_generated_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive group means",
            "",
            "| Condition | Human label | n | Mean judge score |",
            "|---|---|---:|---:|",
        ]
    )
    for (condition, label), values in sorted(grouped.items()):
        lines.append(f"| {condition} | {label} | {len(values)} | {sum(values)/len(values):.6f} |")
    lines.extend(
        [
            "",
            "## Descriptive coherence strata",
            "",
            "| Condition | Coherent | n | Mean judge score |",
            "|---|---:|---:|---:|",
        ]
    )
    for (condition, coherent), values in sorted(grouped_coherence.items()):
        lines.append(
            f"| {condition} | {str(coherent).lower()} | {len(values)} | "
            f"{sum(values)/len(values):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    summary = {
        "schema_version": 1,
        "status": "complete_descriptive_no_posthoc_threshold",
        "rows": len(merged),
        "key_sha256": sha256_file(key_path),
        "human_labels_sha256": sha256_file(human_path),
        "score_sha256_by_condition": {
            condition: sha256_file(path) for condition, path in score_paths.items()
        },
        "table_json_sha256": sha256_file(json_path),
        "table_csv_sha256": sha256_file(csv_path),
        "table_markdown_sha256": sha256_file(md_path),
        "interpretation": (
            "qualitative sanity table only; judge scores are continuous and the "
            "12-prompt-per-condition audit is not inferential"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quantify period-two structure in the post-hoc latent token readout."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from mats_latent_safety.hashing import sha256_file


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "artifacts/discovery/results/posthoc_token_mode_readout/"
            "token_mode_readout.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/discovery/results/posthoc_token_mode_readout/"
            "period_two_analysis.json"
        ),
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite periodicity analysis")
    payload = json.loads(input_path.read_text())
    if payload["status"] != "complete":
        raise ValueError("readout is incomplete")

    tasks = {}
    for task in sorted({row["task"] for row in payload["rows"]}):
        prompt_rows = [row for row in payload["rows"] if row["task"] == task]
        pooled = {"same_parity": [], "opposite_parity": [], "lag_1": [], "lag_2": []}
        per_prompt = []
        for row in prompt_rows:
            token_sets = [
                {token["token_id"] for token in readout["top_tokens"]}
                for readout in row["readouts"]
            ]
            prompt_values = {"same_parity": [], "opposite_parity": []}
            for left, right in itertools.combinations(range(len(token_sets)), 2):
                overlap = jaccard(token_sets[left], token_sets[right])
                parity_key = "same_parity" if (left - right) % 2 == 0 else "opposite_parity"
                pooled[parity_key].append(overlap)
                prompt_values[parity_key].append(overlap)
                lag = abs(left - right)
                if lag in (1, 2):
                    pooled[f"lag_{lag}"].append(overlap)
            same = statistics.mean(prompt_values["same_parity"])
            opposite = statistics.mean(prompt_values["opposite_parity"])
            per_prompt.append(
                {
                    "prompt_id": row["prompt_id"],
                    "same_parity_mean_top10_jaccard": same,
                    "opposite_parity_mean_top10_jaccard": opposite,
                    "same_minus_opposite": same - opposite,
                }
            )
        summary = {key: statistics.mean(values) for key, values in pooled.items()}
        summary["same_minus_opposite"] = (
            summary["same_parity"] - summary["opposite_parity"]
        )
        summary["lag2_minus_lag1"] = summary["lag_2"] - summary["lag_1"]
        tasks[task] = {"summary": summary, "per_prompt": per_prompt}

    result = {
        "schema_version": 1,
        "status": "complete",
        "analysis_role": "posthoc_descriptive_periodicity_check",
        "metric": "jaccard_overlap_of_native_top10_token_ids",
        "depth_indexing": "one_through_six",
        "hypothesis_checked_after_viewing_readout": True,
        "period_two_structure_observed": {
            task: values["summary"]["same_minus_opposite"] > 0
            and values["summary"]["lag2_minus_lag1"] > 0
            for task, values in tasks.items()
        },
        "c_thought": 2,
        "causal_attribution_to_c_thought": False,
        "causal_limit": (
            "only c_thought=2 was trained; no alternate-c_thought control exists, "
            "and task serializations differ"
        ),
        "source_sha256": sha256_file(input_path),
        "tasks": tasks,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute the preregistered paired-prompt endpoint comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from mats_latent_safety.hashing import sha256_file
from mats_latent_safety.runtime import git_revision, slurm_job_id


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official_safety_scoring.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config["status"] != "authorized_after_blind_human_labels_committed":
        raise ValueError("official scoring comparison is not authorized")

    conditions = ("cot_u3", "coco_u3_k6")
    results = {
        condition: json.loads(Path(config["conditions"][condition]["scores"]).read_text())
        for condition in conditions
    }
    for condition, result in results.items():
        if result.get("status") != "passed" or result.get("official_condition") != condition:
            raise ValueError(f"invalid official score result for {condition}")
    by_condition = {
        condition: {row["prompt_id"]: float(row["score"]) for row in result["records"]}
        for condition, result in results.items()
    }
    ids = list(by_condition["cot_u3"])
    if len(ids) != 60 or set(ids) != set(by_condition["coco_u3_k6"]):
        raise ValueError("official score files do not contain the same 60 prompts")
    differences = [
        by_condition["coco_u3_k6"][prompt_id] - by_condition["cot_u3"][prompt_id]
        for prompt_id in ids
    ]
    comparison = config["comparison"]
    rng = random.Random(int(comparison["bootstrap_seed"]))
    replicates = []
    for _ in range(int(comparison["bootstrap_samples"])):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        replicates.append(sum(sample) / len(sample))
    delta = sum(differences) / len(differences)
    interval = [percentile(replicates, 0.025), percentile(replicates, 0.975)]
    margin = float(comparison["frozen_equivalence_margin"])
    result = {
        "schema_version": 1,
        "status": "complete",
        "effect": comparison["effect"],
        "cot_u3_mean": results["cot_u3"]["mean_score"],
        "coco_u3_k6_mean": results["coco_u3_k6"]["mean_score"],
        "mean_delta": delta,
        "paired_prompt_bootstrap_95_ci": interval,
        "bootstrap_seed": int(comparison["bootstrap_seed"]),
        "bootstrap_samples": int(comparison["bootstrap_samples"]),
        "frozen_equivalence_margin": margin,
        "ci_entirely_within_equivalence_margin": interval[0] > -margin and interval[1] < margin,
        "prompt_uncertainty_only": True,
        "training_seed_uncertainty_claimed": False,
        "score_sha256_by_condition": {
            condition: sha256_file(Path(config["conditions"][condition]["scores"]))
            for condition in conditions
        },
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "paired_records": [
            {
                "prompt_id": prompt_id,
                "cot_u3_score": by_condition["cot_u3"][prompt_id],
                "coco_u3_k6_score": by_condition["coco_u3_k6"][prompt_id],
                "delta": by_condition["coco_u3_k6"][prompt_id]
                - by_condition["cot_u3"][prompt_id],
            }
            for prompt_id in ids
        ],
    }
    output = Path(comparison["output"])
    if output.exists():
        raise FileExistsError("refusing to overwrite official paired comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


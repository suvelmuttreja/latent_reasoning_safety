#!/usr/bin/env python3
"""Score terminating official responses and bound nonterminating outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from huggingface_hub import model_info
from strong_reject.evaluate import strongreject_finetuned

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.official_eval import partition_official_safety_rows_for_bounds
from mats_latent_safety.runtime import git_revision, slurm_job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    parser.add_argument("--official-condition", required=True)
    parser.add_argument("--authorization-config", required=True)
    args = parser.parse_args()

    authorization = yaml.safe_load(Path(args.authorization_config).read_text())
    if authorization.get("status") != "authorized_after_blind_human_labels_committed":
        raise ValueError("official bounded scoring is not authorized")
    condition = authorization.get("conditions", {}).get(args.official_condition, {})
    if condition.get("scoring_status") != "authorized_bounds_only":
        raise ValueError("condition is not authorized for bounds-only scoring")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite official bounded scores")
    generation_path = Path(args.generations)
    generation_rows = [
        json.loads(line) for line in generation_path.read_text().splitlines() if line.strip()
    ]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())["records"]
    evaluation_path = Path(args.evaluation_config)
    evaluation = yaml.safe_load(evaluation_path.read_text())
    completed, incomplete = partition_official_safety_rows_for_bounds(
        generation_rows, manifest, args.official_condition
    )
    manifest_by_id = {row["id"]: row for row in manifest}

    evaluator = evaluation["evaluator"]
    batch = {
        "forbidden_prompt": [
            manifest_by_id[row["prompt_id"]]["forbidden_prompt"] for row in completed
        ],
        "response": [row["parsed_final_answer"] for row in completed],
    }
    judge_max_response_length = int(evaluator["max_response_length"])
    score_result = strongreject_finetuned(batch, max_response_length=judge_max_response_length)
    scores = [float(score) for score in score_result["score"]]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise RuntimeError(f"invalid scores: {scores}")
    judge = model_info(evaluator["judge_model"], revision=evaluator["judge_revision"])
    if judge.sha != evaluator["judge_revision"]:
        raise ValueError("resolved judge revision differs from frozen evaluation config")

    records = []
    for generation, score in zip(completed, scores):
        manifest_row = manifest_by_id[generation["prompt_id"]]
        payload = {
            "forbidden_prompt": manifest_row["forbidden_prompt"],
            "response": generation["parsed_final_answer"],
        }
        records.append(
            {
                "prompt_id": generation["prompt_id"],
                "category": manifest_row["category"],
                "evaluator_payload_sha256": sha256_json(payload),
                "score": score,
                "final_answer_characters": len(generation["parsed_final_answer"]),
                "full_answer_tokens": generation.get("answer_tokens"),
                "full_generated_tokens": generation.get("generated_tokens"),
            }
        )
    n = len(manifest)
    score_sum = sum(scores)
    lower = score_sum / n
    upper = (score_sum + len(incomplete)) / n
    result = {
        "schema_version": 1,
        "status": "partial_identification_bounds",
        "point_estimate_available": False,
        "official_condition": args.official_condition,
        "estimand": "full_manifest_mean_under_extreme_scoring_of_nonterminating_prompts",
        "lower_bound_all_nonterminations_zero": lower,
        "upper_bound_all_nonterminations_one": upper,
        "bound_width": upper - lower,
        "manifest_rows": n,
        "eos_completed_rows": len(completed),
        "nonterminating_rows": len(incomplete),
        "nontermination_rate": len(incomplete) / n,
        "nonterminating_prompt_ids": [row["prompt_id"] for row in incomplete],
        "completed_case_mean_descriptive_only": score_sum / len(scores),
        "evaluator": "strongreject_finetuned",
        "strong_reject_git_revision": evaluator["git_revision"],
        "judge_model": judge.id,
        "judge_revision": judge.sha,
        "judge_max_response_length": judge_max_response_length,
        "primary_payload": "parsed_final_answer_only_for_eos_completed_rows",
        "generations_sha256": sha256_file(generation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

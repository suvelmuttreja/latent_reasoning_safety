#!/usr/bin/env python3
"""Score cached final answers without regenerating model outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from huggingface_hub import model_info
from strong_reject.evaluate import strongreject_finetuned

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.official_eval import validate_official_safety_rows
from mats_latent_safety.runtime import git_revision, slurm_job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    parser.add_argument("--official-condition")
    parser.add_argument("--authorization-config")
    args = parser.parse_args()
    if args.authorization_config:
        authorization = yaml.safe_load(Path(args.authorization_config).read_text())
        if authorization.get("status") != "authorized_after_blind_human_labels_committed":
            raise ValueError("official automatic scoring is not authorized")
    output = Path(args.output)
    if args.official_condition and output.exists():
        raise FileExistsError("refusing to overwrite official automatic scores")
    generation_path = Path(args.generations)
    generation_rows = [
        row
        for line in Path(args.generations).read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    ]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())["records"]
    evaluation_path = Path(args.evaluation_config)
    evaluation = yaml.safe_load(evaluation_path.read_text())
    if args.official_condition:
        max_incomplete_rate = float(evaluation["endpoint_truncation_guard"]["threshold"])
        selected = validate_official_safety_rows(
            generation_rows,
            manifest,
            args.official_condition,
            max_incomplete_rate=max_incomplete_rate,
        )
    else:
        generations = {row["prompt_id"]: row for row in generation_rows}
        if len(generations) != len(generation_rows):
            raise ValueError("cached generations contain duplicate prompt IDs")
        missing = [row["id"] for row in manifest if row["id"] not in generations]
        if missing:
            raise ValueError(f"cached generations missing manifest prompts: {missing}")
        selected = [generations[row["id"]] for row in manifest]
    if not args.official_condition and any(row["truncated"] for row in selected):
        raise ValueError(
            "refusing to judge truncated calibration outputs; finish cap calibration first"
        )
    evaluator = evaluation["evaluator"]
    batch = {
        "forbidden_prompt": [row["forbidden_prompt"] for row in manifest],
        "response": [row["parsed_final_answer"] for row in selected],
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
    for manifest_row, generation, score in zip(manifest, selected, scores):
        payload = {
            "forbidden_prompt": manifest_row["forbidden_prompt"],
            "response": generation["parsed_final_answer"],
        }
        records.append(
            {
                "prompt_id": manifest_row["id"],
                "category": manifest_row["category"],
                "generation_code_revision": generation["code_revision"],
                "evaluator_payload_sha256": sha256_json(payload),
                "score": score,
                "final_answer_characters": len(generation["parsed_final_answer"]),
                "full_answer_tokens": generation.get("answer_tokens"),
                "full_generated_tokens": generation.get("generated_tokens"),
            }
        )
    result = {
        "schema_version": 1,
        "status": "passed",
        "evaluator": "strongreject_finetuned",
        "strong_reject_git_revision": evaluator["git_revision"],
        "judge_model": judge.id,
        "judge_revision": judge.sha,
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "official_condition": args.official_condition,
        "generations_sha256": sha256_file(generation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "primary_payload": "parsed_final_answer_only",
        "judge_max_response_length": judge_max_response_length,
        "judge_max_response_length_unit": "judge_tokenizer_tokens",
        "judge_truncation_side": "left_keep_rightmost_tokens",
        "mean_score": sum(scores) / len(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

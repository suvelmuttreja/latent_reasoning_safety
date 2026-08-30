#!/usr/bin/env python3
"""Score cached final answers without regenerating model outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from huggingface_hub import model_info
from strong_reject.evaluate import strongreject_finetuned

from mats_latent_safety.hashing import sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generations = {
        row["prompt_id"]: row
        for line in Path(args.generations).read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    manifest = json.loads(Path(args.manifest).read_text())["records"]
    missing = [row["id"] for row in manifest if row["id"] not in generations]
    if missing:
        raise ValueError(f"cached generations missing manifest prompts: {missing}")
    selected = [generations[row["id"]] for row in manifest]
    if any(row["truncated"] for row in selected):
        raise ValueError("refusing to judge truncated calibration outputs; finish cap calibration first")
    batch = {
        "forbidden_prompt": [row["forbidden_prompt"] for row in manifest],
        "response": [row["parsed_final_answer"] for row in selected],
    }
    judge_max_response_length = 512
    score_result = strongreject_finetuned(
        batch, max_response_length=judge_max_response_length
    )
    scores = [float(score) for score in score_result["score"]]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise RuntimeError(f"invalid scores: {scores}")
    judge = model_info("qylu4156/strongreject-15k-v1")
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
            }
        )
    result = {
        "schema_version": 1,
        "status": "passed",
        "evaluator": "strongreject_finetuned",
        "strong_reject_git_revision": "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47",
        "judge_model": judge.id,
        "judge_revision": judge.sha,
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "primary_payload": "parsed_final_answer_only",
        "judge_max_response_length": judge_max_response_length,
        "judge_max_response_length_unit": "judge_tokenizer_tokens",
        "judge_truncation_side": "left_keep_rightmost_tokens",
        "mean_score": sum(scores) / len(scores),
        "minimum_score": min(scores),
        "maximum_score": max(scores),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

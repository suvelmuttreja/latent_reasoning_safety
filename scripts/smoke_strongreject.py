#!/usr/bin/env python3
"""Load and score five cached examples with the registered local evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from huggingface_hub import model_info
from strong_reject.evaluate import strongreject_finetuned

from mats_latent_safety.runtime import git_revision, slurm_job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="configs/strongreject_smoke_cases.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())["cases"]
    if len(cases) != 5:
        raise ValueError("the registered evaluator smoke requires exactly five cases")
    batch = {
        "forbidden_prompt": [case["forbidden_prompt"] for case in cases],
        "response": [case["response"] for case in cases],
    }
    result = strongreject_finetuned(batch)
    scores = [float(score) for score in result["score"]]
    if len(scores) != 5 or not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise RuntimeError(f"invalid evaluator scores: {scores}")
    judge = model_info("qylu4156/strongreject-15k-v1")
    expected_judge_revision = "4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
    if judge.sha != expected_judge_revision:
        raise RuntimeError(
            f"judge revision changed: expected {expected_judge_revision}, resolved {judge.sha}"
        )
    payload = {
        "status": "passed",
        "evaluator": "strongreject_finetuned",
        "strong_reject_git_revision": "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47",
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "judge_model": judge.id,
        "judge_revision": judge.sha,
        "cases": [
            {"id": case["id"], "intended_class": case["intended_class"], "score": score}
            for case, score in zip(cases, scores)
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

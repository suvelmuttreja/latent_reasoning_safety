#!/usr/bin/env python3
"""Score and summarize the paired M0 dual-serialization anchor."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml
from huggingface_hub import model_info
from strong_reject.evaluate import strongreject_finetuned

from mats_latent_safety.hashing import sha256_json
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
    parser.add_argument("--config", default="configs/format_anchor.yaml")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())["records"]
    generations = {
        (row["condition"], row["prompt_id"]): row
        for line in Path(args.generations).read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    ordered = [
        generations[(condition, record["id"])]
        for record in manifest
        for condition in config["conditions"]
    ]
    if len(ordered) != len(manifest) * len(config["conditions"]):
        raise ValueError("format-anchor generations are incomplete")
    prompt_lookup = {row["id"]: row["forbidden_prompt"] for row in manifest}
    payload = {
        "forbidden_prompt": [prompt_lookup[row["prompt_id"]] for row in ordered],
        "response": [row["parsed_final_answer"] for row in ordered],
    }
    evaluated = strongreject_finetuned(payload)
    scores = [float(value) for value in evaluated["score"]]
    if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
        raise RuntimeError(f"invalid scores: {scores}")
    score_lookup = {
        (row["condition"], row["prompt_id"]): score
        for row, score in zip(ordered, scores)
    }
    pairs = []
    for record in manifest:
        native = score_lookup[("native_chat", record["id"])]
        raw = score_lookup[("coconut_raw_question", record["id"])]
        pairs.append(
            {
                "prompt_id": record["id"],
                "category": record["category"],
                "native_chat_score": native,
                "coconut_raw_question_score": raw,
                "raw_minus_native": raw - native,
                "native_payload_sha256": sha256_json(
                    {
                        "forbidden_prompt": record["forbidden_prompt"],
                        "response": generations[("native_chat", record["id"])][
                            "parsed_final_answer"
                        ],
                    }
                ),
                "raw_payload_sha256": sha256_json(
                    {
                        "forbidden_prompt": record["forbidden_prompt"],
                        "response": generations[("coconut_raw_question", record["id"])][
                            "parsed_final_answer"
                        ],
                    }
                ),
            }
        )
    differences = [row["raw_minus_native"] for row in pairs]
    rng = random.Random(int(config["bootstrap"]["seed"]))
    replicates = []
    for _ in range(int(config["bootstrap"]["samples"])):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        replicates.append(sum(sample) / len(sample))
    judge = model_info(config["evaluator"]["judge_model"])
    result = {
        "schema_version": 1,
        "status": "passed_pending_transcript_review_and_format_freeze",
        "interpretation": "M0 serialization anchor only; never subtract from checkpoint scores",
        "effect_sign": config["effect_sign"],
        "native_chat_mean": sum(row["native_chat_score"] for row in pairs) / len(pairs),
        "coconut_raw_question_mean": sum(
            row["coconut_raw_question_score"] for row in pairs
        )
        / len(pairs),
        "mean_raw_minus_native": sum(differences) / len(differences),
        "paired_bootstrap_95_ci": [percentile(replicates, 0.025), percentile(replicates, 0.975)],
        "frozen_equivalence_margin": abs(sum(differences) / len(differences)),
        "truncated_generations": sum(bool(row["truncated"]) for row in ordered),
        "evaluator": config["evaluator"]["name"],
        "judge_model": judge.id,
        "judge_revision": judge.sha,
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "pairs": pairs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

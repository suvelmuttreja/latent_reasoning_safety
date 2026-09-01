#!/usr/bin/env python3
"""Report paired prompt uncertainty and observed spread for two CoT-u1 seeds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from mats_latent_safety.hashing import sha256_file
from mats_latent_safety.runtime import git_revision


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_scores(path: Path, condition: str) -> tuple[dict, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed" or payload.get("official_condition") != condition:
        raise ValueError(f"invalid score artifact for {condition}")
    by_id = {row["prompt_id"]: float(row["score"]) for row in payload["records"]}
    if len(by_id) != 60 or len(payload["records"]) != 60:
        raise ValueError(f"{condition} must contain exactly 60 unique prompt scores")
    observed_mean = sum(by_id.values()) / len(by_id)
    if abs(observed_mean - float(payload["mean_score"])) > 1e-12:
        raise ValueError(f"stored mean differs from records for {condition}")
    return payload, by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m0",
        default="artifacts/discovery/results/official_safety/scores/m0.json",
    )
    parser.add_argument(
        "--seed42",
        default="artifacts/discovery/results/dense_safety/scores/cot_u1.json",
    )
    parser.add_argument(
        "--seed43",
        default=(
            "artifacts/discovery/results/matched_4b_cot_seed43/"
            "scores/cot_seed43_u1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/discovery/results/matched_4b_cot_seed43/"
            "cot_u1_seed_comparison.json"
        ),
    )
    parser.add_argument(
        "--coconut-bounds",
        default=(
            "artifacts/discovery/results/dense_safety/scores/"
            "coco_u1_k2_bounds.json"
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    paths = {
        "m0": Path(args.m0),
        "seed42": Path(args.seed42),
        "seed43": Path(args.seed43),
    }
    expected_conditions = {"m0": "m0", "seed42": "cot_u1", "seed43": "cot_seed43_u1"}
    payloads = {}
    scores = {}
    for label, path in paths.items():
        payloads[label], scores[label] = load_scores(path, expected_conditions[label])

    prompt_ids = list(scores["m0"])
    if any(set(values) != set(prompt_ids) for values in scores.values()):
        raise ValueError("score artifacts do not contain the same prompt IDs")

    coconut_path = Path(args.coconut_bounds)
    coconut = json.loads(coconut_path.read_text())
    if (
        coconut.get("status") != "partial_identification_bounds"
        or coconut.get("official_condition") != "coco_u1_k2"
    ):
        raise ValueError("invalid Coconut-u1 bounds artifact")
    coconut_by_id = {
        row["prompt_id"]: float(row["score"]) for row in coconut["records"]
    }
    if not set(coconut_by_id).issubset(prompt_ids):
        raise ValueError("Coconut bounds contain an unknown prompt ID")
    missing_coconut = [key for key in prompt_ids if key not in coconut_by_id]
    if len(missing_coconut) != int(coconut["nonterminating_rows"]):
        raise ValueError("Coconut missing IDs differ from its nontermination count")

    contributions = {
        "seed42_minus_m0": [scores["seed42"][key] - scores["m0"][key] for key in prompt_ids],
        "seed43_minus_m0": [scores["seed43"][key] - scores["m0"][key] for key in prompt_ids],
        "seed43_minus_seed42": [
            scores["seed43"][key] - scores["seed42"][key] for key in prompt_ids
        ],
    }
    for seed_label in ("seed42", "seed43"):
        contributions[f"{seed_label}_minus_coconut_lower"] = [
            scores[seed_label][key] - coconut_by_id.get(key, 1.0) for key in prompt_ids
        ]
        contributions[f"{seed_label}_minus_coconut_upper"] = [
            scores[seed_label][key] - coconut_by_id.get(key, 0.0) for key in prompt_ids
        ]
    rng = random.Random(args.bootstrap_seed)
    replicates = {effect: [] for effect in contributions}
    for _ in range(args.bootstrap_samples):
        indices = [rng.randrange(len(prompt_ids)) for _ in prompt_ids]
        for effect, values in contributions.items():
            replicates[effect].append(sum(values[index] for index in indices) / len(indices))

    effects = {
        effect: {
            "mean_delta": sum(values) / len(values),
            "paired_prompt_bootstrap_95_ci": [
                percentile(replicates[effect], 0.025),
                percentile(replicates[effect], 0.975),
            ],
        }
        for effect, values in contributions.items()
    }
    bounded_substrate_contrasts = {}
    for seed_label in ("seed42", "seed43"):
        lower = effects.pop(f"{seed_label}_minus_coconut_lower")
        upper = effects.pop(f"{seed_label}_minus_coconut_upper")
        bounded_substrate_contrasts[seed_label] = {
            "effect": f"cot_{seed_label}_u1_minus_coco_u1_k2",
            "identified_set": [lower["mean_delta"], upper["mean_delta"]],
            "lower_endpoint_bootstrap_95_ci": lower[
                "paired_prompt_bootstrap_95_ci"
            ],
            "upper_endpoint_bootstrap_95_ci": upper[
                "paired_prompt_bootstrap_95_ci"
            ],
            "conservative_identified_set_95_confidence_region": [
                lower["paired_prompt_bootstrap_95_ci"][0],
                upper["paired_prompt_bootstrap_95_ci"][1],
            ],
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "analysis_role": "two_seed_cot_u1_noise_floor_diagnostic",
        "prompts": len(prompt_ids),
        "means": {label: payloads[label]["mean_score"] for label in payloads},
        "effects": effects,
        "coconut_nonterminating_prompt_ids": missing_coconut,
        "bounded_substrate_contrasts": bounded_substrate_contrasts,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "prompt_uncertainty_only": True,
        "seed_confidence_interval_claimed": False,
        "coconut_seed_variability_claimed": False,
        "binary_replication_rule": None,
        "report_result_regardless_of_direction": True,
        "score_sha256": {label: sha256_file(path) for label, path in paths.items()},
        "coconut_bounds_sha256": sha256_file(coconut_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "code_revision": git_revision(),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite seed comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

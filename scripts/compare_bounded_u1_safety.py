#!/usr/bin/env python3
"""Bootstrap the paired CoT-versus-Coconut u1 partially identified contrast."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cot-scores",
        default="artifacts/discovery/results/dense_safety/scores/cot_u1.json",
    )
    parser.add_argument(
        "--coconut-bounds",
        default="artifacts/discovery/results/dense_safety/scores/coco_u1_k2_bounds.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/discovery/results/dense_safety/trajectory/"
            "cot_minus_coconut_u1_bounded_bootstrap.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    cot_path = Path(args.cot_scores)
    coconut_path = Path(args.coconut_bounds)
    cot = json.loads(cot_path.read_text())
    coconut = json.loads(coconut_path.read_text())
    if cot.get("official_condition") != "cot_u1" or cot.get("status") != "passed":
        raise ValueError("invalid cot_u1 score artifact")
    if (
        coconut.get("official_condition") != "coco_u1_k2"
        or coconut.get("status") != "partial_identification_bounds"
    ):
        raise ValueError("invalid coco_u1_k2 bounds artifact")

    cot_by_id = {row["prompt_id"]: float(row["score"]) for row in cot["records"]}
    coconut_by_id = {
        row["prompt_id"]: float(row["score"]) for row in coconut["records"]
    }
    prompt_ids = list(cot_by_id)
    if len(prompt_ids) != 60 or len(set(prompt_ids)) != 60:
        raise ValueError("cot_u1 must contain exactly 60 unique prompts")
    if not set(coconut_by_id).issubset(cot_by_id):
        raise ValueError("Coconut bounds contain an unknown prompt ID")
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in coconut_by_id]
    if len(missing) != int(coconut["nonterminating_rows"]):
        raise ValueError("Coconut missing IDs differ from its nontermination count")

    # Effect orientation is CoT minus Coconut. Assigning missing Coconut scores
    # one gives the lower endpoint; assigning zero gives the upper endpoint.
    lower_contributions = [
        cot_by_id[prompt_id] - coconut_by_id.get(prompt_id, 1.0)
        for prompt_id in prompt_ids
    ]
    upper_contributions = [
        cot_by_id[prompt_id] - coconut_by_id.get(prompt_id, 0.0)
        for prompt_id in prompt_ids
    ]
    rng = random.Random(args.seed)
    lower_replicates = []
    upper_replicates = []
    for _ in range(args.samples):
        indices = [rng.randrange(len(prompt_ids)) for _ in prompt_ids]
        lower_replicates.append(
            sum(lower_contributions[index] for index in indices) / len(indices)
        )
        upper_replicates.append(
            sum(upper_contributions[index] for index in indices) / len(indices)
        )

    lower = sum(lower_contributions) / len(lower_contributions)
    upper = sum(upper_contributions) / len(upper_contributions)
    lower_ci = [percentile(lower_replicates, 0.025), percentile(lower_replicates, 0.975)]
    upper_ci = [percentile(upper_replicates, 0.025), percentile(upper_replicates, 0.975)]
    confidence_region = [lower_ci[0], upper_ci[1]]
    result = {
        "schema_version": 1,
        "status": "complete",
        "effect": "cot_u1_minus_coco_u1_k2",
        "estimand": "paired_prompt_mean_with_coconut_nonterminations_bounded_in_zero_one",
        "prompts": len(prompt_ids),
        "coconut_nonterminating_prompt_ids": missing,
        "identified_set": [lower, upper],
        "lower_endpoint_bootstrap_95_ci": lower_ci,
        "upper_endpoint_bootstrap_95_ci": upper_ci,
        "conservative_identified_set_95_confidence_region": confidence_region,
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.samples,
        "prompt_uncertainty_only": True,
        "training_seed_uncertainty_claimed": False,
        "sign_identified_in_pointwise_bounds": lower > 0,
        "sign_identified_with_prompt_uncertainty": confidence_region[0] > 0,
        "interpretation": (
            "suggestive_transient_substrate_structure_not_detected_interaction"
            if confidence_region[0] <= 0
            else "transient_substrate_interaction_detected"
        ),
        "cot_scores_sha256": sha256_file(cot_path),
        "coconut_bounds_sha256": sha256_file(coconut_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "code_revision": git_revision(),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite bounded u1 comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

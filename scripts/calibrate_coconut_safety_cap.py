#!/usr/bin/env python3
"""Calibrate a branch safety-response cap from token lengths without judging."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from gate_4b_coco_u1 import generate, read_manifest, write_jsonl
from mats_latent_safety.cap_calibration import select_smallest_cap_or_none
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import ensure_latent_tokens
from train_4b_skip0_stage import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coconut_safety_cap_calibration.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-generations")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    evaluation_path = Path(config["evaluation_config"])
    evaluation = yaml.safe_load(evaluation_path.read_text())
    train_path = Path(config["train_config"])
    train = yaml.safe_load(train_path.read_text())
    frozen = evaluation["sampling"]
    for key in ("do_sample", "temperature", "top_p", "top_k"):
        if config["sampling"][key] != frozen[key]:
            raise ValueError(f"calibration {key} differs from frozen sampler")
    if config["sampling"]["stop_tokens"] != evaluation["coconut_generation"]["stop_tokens"]:
        raise ValueError("calibration stop tokens differ from frozen evaluation")

    checkpoint_dir = Path(args.checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    checkpoint = checkpoint_dir / "model_state.pt"
    checkpoint_hash = sha256_file(checkpoint)
    registered = config["checkpoint"]
    expected_branch = registered.get("branch", "coconut_skip0")
    if metadata["branch"] != expected_branch:
        raise ValueError("checkpoint branch differs from calibration config")
    if metadata["completed_stage"] != registered["stage"] or metadata["k"] != registered["k"]:
        raise ValueError("checkpoint stage/K differs from calibration config")
    if checkpoint_hash != registered["model_state_sha256"]:
        raise ValueError("checkpoint hash differs from registered calibration checkpoint")
    if metadata["model_state_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint hash differs from stage metadata")
    if metadata["durability"]["status"] != "model_and_tokenizer_uploaded":
        raise ValueError("checkpoint lacks durable model/tokenizer evidence")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite safety-cap calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    code_revision = git_revision()
    manifest_path = Path(config["manifest"])
    records = read_manifest(manifest_path)
    settings = {
        "serialization": config["serialization"],
        "max_new_tokens": config["calibration_max_new_tokens"],
        **config["sampling"],
    }
    if "generation_scaffold" in config:
        settings["scaffold_kind"] = config["generation_scaffold"]
    if settings.get("scaffold_kind") == "explicit_cot" and config["k_values"] != [0]:
        raise ValueError("explicit-CoT calibration requires exactly k_values: [0]")
    prior_rows = []
    prior_by_key = {}
    prior_path = Path(args.prior_generations) if args.prior_generations else None
    continuation = config.get("continuation")
    if (prior_path is None) != (continuation is None):
        raise ValueError("continuation config and --prior-generations must be used together")
    if prior_path is not None:
        if sha256_file(prior_path) != continuation["parent_generations_sha256"]:
            raise ValueError("prior calibration hash differs from continuation config")
        prior_rows = [json.loads(line) for line in prior_path.read_text().splitlines()]
        prior_by_key = {(int(row["k"]), row["prompt_id"]): row for row in prior_rows}
        expected_keys = {(int(k), record["id"]) for k in config["k_values"] for record in records}
        if set(prior_by_key) != expected_keys or len(prior_rows) != len(expected_keys):
            raise ValueError("prior calibration rows do not match the frozen K/prompt grid")
        if any(
            row["evaluator_payload"] is not None or row["evaluator_score"] is not None
            for row in prior_rows
        ):
            raise ValueError("prior calibration unexpectedly contains evaluator data")
        observed_length_stops = {
            str(k): sum(
                int(row["k"]) == int(k) and row["stop_reason"] == "length" for row in prior_rows
            )
            for k in config["k_values"]
        }
        if observed_length_stops != continuation["parent_length_stops_by_k"]:
            raise ValueError("prior length-stop counts differ from continuation config")

    tokenizer = AutoTokenizer.from_pretrained(train["model_id"], revision=train["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    model = load_model(train, tokenizer, markers, torch.device("cuda"))
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    reused_rows = 0
    extended_rows = 0
    decoded_prefix_matches = 0
    for k in config["k_values"]:
        for index, record in enumerate(records):
            prior = prior_by_key.get((int(k), record["id"]))
            if prior is not None:
                if (
                    prior["prompt_sha256"] != record["sha256"]
                    or prior["model_id"] != train["model_id"]
                    or prior["model_revision"] != train["model_revision"]
                    or prior["serialization"] != config["serialization"]
                ):
                    raise ValueError("prior calibration provenance differs from continuation")
                if prior["stop_reason"] != "length":
                    reused = dict(prior)
                    reused["calibration_source"] = "prior_completed_generation"
                    rows.append(reused)
                    reused_rows += 1
                    continue
            result = generate(
                model,
                tokenizer,
                markers,
                record["forbidden_prompt"],
                int(k),
                settings,
                int(config["seed"]) + index,
            )
            row = {
                "prompt_id": record["id"],
                "prompt_sha256": record["sha256"],
                "category": record["category"],
                "model_id": train["model_id"],
                "model_revision": train["model_revision"],
                "code_revision": code_revision,
                "k": int(k),
                "generation_config_sha256": sha256_json(settings),
                "evaluator_payload": None,
                "evaluator_score": None,
                **result,
            }
            if prior is not None:
                prefix_matches = result["raw_output"].startswith(prior["raw_output"])
                decoded_prefix_matches += int(prefix_matches)
                row.update(
                    {
                        "calibration_source": "extended_prior_length_stop",
                        "prior_generated_tokens": prior["generated_tokens"],
                        "prior_generation_config_sha256": prior["generation_config_sha256"],
                        "decoded_prefix_matches_prior": prefix_matches,
                    }
                )
                extended_rows += 1
            else:
                row["calibration_source"] = "initial_generation"
            rows.append(row)
    rows_path = output_dir / "generations.jsonl"
    write_jsonl(rows_path, rows)
    selected, projection = select_smallest_cap_or_none(
        rows,
        [int(value) for value in config["candidate_caps"]],
        float(config["selection_rule"]["threshold"]),
    )
    by_k = {}
    for k in config["k_values"]:
        group = [row for row in rows if int(row["k"]) == int(k)]
        lengths = sorted(int(row["generated_tokens"]) for row in group)
        by_k[str(k)] = {
            "outputs": len(group),
            "eos_stops": sum(row["stop_reason"] == "eos_token" for row in group),
            "calibration_cap_hits": sum(
                row["stop_reason"] == "length"
                or row["generated_tokens"] >= config["calibration_max_new_tokens"]
                for row in group
            ),
            "mean_tokens": statistics.mean(lengths),
            "p95_tokens": lengths[max(0, int(0.95 * len(lengths)) - 1)],
            "max_tokens": max(lengths),
        }
    passed = selected is not None
    summary = {
        "schema_version": 1,
        "status": (
            "length_calibration_complete_pending_cap_freeze_commit"
            if passed
            else "no_registered_cap_passed_official_scoring_remains_blocked"
        ),
        "selected_cap": selected,
        "selection_used_only_lengths_and_stop_reasons": True,
        "evaluator_loaded": False,
        "slurm_job_id": slurm_job_id(),
        "code_revision": code_revision,
        "checkpoint_sha256": checkpoint_hash,
        "strict_load_missing": list(load_result.missing_keys),
        "strict_load_unexpected": list(load_result.unexpected_keys),
        "config_sha256": sha256_file(config_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generations_sha256": sha256_file(rows_path),
        "continuation": {
            "prior_generations_sha256": (
                sha256_file(prior_path) if prior_path is not None else None
            ),
            "prior_rows_reused": reused_rows,
            "prior_length_stops_extended": extended_rows,
            "decoded_prefix_matches": decoded_prefix_matches,
        },
        "by_k": by_k,
        "candidate_projection": projection,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

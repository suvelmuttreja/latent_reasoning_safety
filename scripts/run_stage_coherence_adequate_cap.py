#!/usr/bin/env python3
"""Run only stage-1 K=0/K=2 coherence at the frozen explicit-thinking cap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from gate_4b_coco_u1 import evaluate_coherence, read_manifest, write_jsonl
from mats_latent_safety.coherence_audit import validate_control_against_frozen_evaluation
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import ensure_latent_tokens
from train_4b_skip0_stage import load_model


def summarize(rows: list[dict], max_new_tokens: int) -> dict:
    def one(group: list[dict]) -> dict:
        return {
            "outputs": len(group),
            "cap_hits": sum(
                row["stop_reason"] == "length"
                or row["generated_tokens"] >= max_new_tokens
                for row in group
            ),
            "missing_closing_think": sum("</think>" not in row["raw_output"] for row in group),
            "eos_stops": sum(row["stop_reason"] == "eos_token" for row in group),
            "mean_generated_tokens": statistics.mean(
                row["generated_tokens"] for row in group
            ),
        }

    return {
        "overall": one(rows),
        "by_k": {
            str(k): one([row for row in rows if int(row["k"]) == k])
            for k in sorted({int(row["k"]) for row in rows})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_4b_coherence_adequate_cap.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    evaluation_path = Path(config["evaluation_config"])
    evaluation = yaml.safe_load(evaluation_path.read_text())
    validate_control_against_frozen_evaluation(config, evaluation)
    train_config_path = Path(config["train_config"])
    train_config = yaml.safe_load(train_config_path.read_text())
    checkpoint_dir = Path(args.checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    checkpoint_path = checkpoint_dir / "model_state.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if metadata["completed_stage"] != config["checkpoint_stage"]:
        raise ValueError("checkpoint stage differs from adequate-cap config")
    if metadata["k"] != config["checkpoint_k"]:
        raise ValueError("checkpoint K differs from adequate-cap config")
    if metadata["model_state_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint hash differs from stage metadata")
    if metadata["config_sha256"] != sha256_json(train_config):
        raise ValueError("checkpoint training config hash differs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "stage_coherence_generations.jsonl"
    summary_path = output_dir / "stage_coherence_summary.json"
    if generations_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite adequate-cap stage coherence output")
    tokenizer = AutoTokenizer.from_pretrained(
        train_config["model_id"], revision=train_config["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    device = torch.device("cuda")
    model = load_model(train_config, tokenizer, markers, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()
    manifest_path = Path(config["manifest"])
    rows = evaluate_coherence(
        model,
        tokenizer,
        markers,
        read_manifest(manifest_path),
        config["stage_coherence"],
        int(config["seed_schedule"]["base_seed"]),
        train_config["model_id"],
        train_config["model_revision"],
    )
    write_jsonl(generations_path, rows)
    summary = {
        "schema_version": 1,
        "status": "technical_generation_complete_pending_combined_blind_review",
        "next_stage_authorized": False,
        "slurm_job_id": slurm_job_id(),
        "code_revision": git_revision(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_matches_stage_metadata": True,
        "strict_load_missing": list(load_result.missing_keys),
        "strict_load_unexpected": list(load_result.unexpected_keys),
        "config_sha256": sha256_json(config),
        "train_config_sha256": sha256_file(train_config_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generations_sha256": sha256_file(generations_path),
        "max_new_tokens": int(config["stage_coherence"]["max_new_tokens"]),
        "summary": summarize(rows, int(config["stage_coherence"]["max_new_tokens"])),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

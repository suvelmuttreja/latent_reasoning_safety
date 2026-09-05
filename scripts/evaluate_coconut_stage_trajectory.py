#!/usr/bin/env python3
"""Evaluate Coconut capability/coherence at K=0 and the stage endpoint K."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from gate_4b_coco_u1 import (
    evaluate_capability,
    evaluate_coherence,
    read_manifest,
    write_jsonl,
)
from mats_latent_safety.constants import END_LATENT, LATENT, START_LATENT
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import ensure_latent_tokens
from mats_latent_safety.trajectory import (
    stage_k_values,
    summarize_generation_rows,
    validate_trajectory_config,
)
from train_4b_skip0_stage import load_model


def enrich_rows(
    rows: list[dict],
    tokenizer,
    settings: dict,
    code_revision: str,
    model_id: str,
    model_revision: str,
    prompt_hashes: dict[str, str],
    evaluation_target: str,
) -> None:
    generation_hash = sha256_json(settings)
    for row in rows:
        prefix = row.pop("raw_serialized_prefix")
        scaffold = prefix + START_LATENT + LATENT * int(row["k"]) + END_LATENT
        row["prompt_sha256"] = prompt_hashes[row["prompt_id"]]
        row["model_id"] = model_id
        row["model_revision"] = model_revision
        row["code_revision"] = code_revision
        row["raw_serialized_input"] = scaffold
        row["input_tokens"] = (
            len(tokenizer.encode(prefix, add_special_tokens=False)) + int(row["k"]) + 2
        )
        row["thinking_tokens"] = len(
            tokenizer.encode(row["parsed_thinking"] or "", add_special_tokens=False)
        )
        row["answer_tokens"] = len(
            tokenizer.encode(row["parsed_final_answer"], add_special_tokens=False)
        )
        row["generation_config_sha256"] = generation_hash
        row["evaluation_target"] = evaluation_target
        if evaluation_target == "gsm8k_exact_match":
            row["evaluator_payload"] = {
                "prediction": row["predicted_answer"],
                "reference": row["reference_answer"],
            }
            row["evaluator_score"] = int(row["correct"])
        else:
            row["evaluator_payload"] = None
            row["evaluator_score"] = None


def export_blind(rows: list[dict], stage: int, output: Path, key_output: Path, seed: int) -> None:
    tagged = [(f"stage{stage}_k{row['k']}", row) for row in rows]
    random.Random(seed).shuffle(tagged)
    blinded, key = [], []
    for index, (condition, row) in enumerate(tagged):
        blind_id = f"stage{stage}-blind-{index:04d}"
        displayed = row["parsed_final_answer"]
        blinded.append({"blind_id": blind_id, "output": displayed, "score_0_to_2": None})
        key.append(
            {
                "blind_id": blind_id,
                "source_condition": condition,
                "prompt_id": row["prompt_id"],
                "output_sha256": hashlib.sha256(displayed.encode()).hexdigest(),
            }
        )
    write_jsonl(output, blinded)
    key_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "shuffle_seed": seed,
                "source_generations_sha256": sha256_file(output.parent / "coherence.jsonl"),
                "records": key,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coconut_stage_trajectory.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capability-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    evaluation_path = Path(config["evaluation_config"])
    evaluation = yaml.safe_load(evaluation_path.read_text())
    validate_trajectory_config(config, evaluation)
    if args.stage not in config["stages"]:
        raise ValueError("stage is not registered for trajectory evaluation")

    train_path = Path(config["train_config"])
    train = yaml.safe_load(train_path.read_text())
    expected_k = stage_k_values(args.stage, int(train["c_thought"]), int(train["max_latent_stage"]))
    checkpoint_dir = Path(args.checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    checkpoint = checkpoint_dir / "model_state.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if metadata["branch"] != "coconut_skip0":
        raise ValueError("trajectory evaluation requires a Coconut checkpoint")
    if metadata["completed_stage"] != args.stage or metadata["k"] != expected_k[1]:
        raise ValueError("checkpoint stage/K differs from the registered trajectory")
    if metadata["model_state_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint hash differs from stage metadata")
    if (
        metadata["model_id"] != train["model_id"]
        or metadata["model_revision"] != train["model_revision"]
    ):
        raise ValueError("checkpoint base model differs from frozen training config")
    if metadata["config_sha256"] != sha256_json(train):
        raise ValueError("checkpoint training config hash differs from frozen config")
    if metadata["durability"]["status"] != "model_and_tokenizer_uploaded":
        raise ValueError("checkpoint lacks durable model/tokenizer upload evidence")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite trajectory output")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(train["model_id"], revision=train["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    model = load_model(train, tokenizer, markers, torch.device("cuda"))
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()
    code_revision = git_revision()

    capability_settings = {**config["capability"], "k_values": expected_k}
    coherence_settings = {**config["coherence"], "k_values": expected_k}
    capability_manifest = read_manifest(Path(config["capability"]["manifest"]))
    coherence_manifest = (
        [] if args.capability_only else read_manifest(Path(config["coherence"]["manifest"]))
    )
    capability = evaluate_capability(
        model,
        tokenizer,
        markers,
        capability_manifest,
        capability_settings,
        config["seed"],
    )
    capability_rows = [row for k in expected_k for row in capability["by_k"][str(k)].pop("outputs")]
    coherence_rows = []
    if not args.capability_only:
        coherence_rows = evaluate_coherence(
            model,
            tokenizer,
            markers,
            coherence_manifest,
            coherence_settings,
            config["seed"],
            train["model_id"],
            train["model_revision"],
        )
    enrich_rows(
        capability_rows,
        tokenizer,
        capability_settings,
        code_revision,
        train["model_id"],
        train["model_revision"],
        {row["id"]: row["sha256"] for row in capability_manifest},
        "gsm8k_exact_match",
    )
    if not args.capability_only:
        enrich_rows(
            coherence_rows,
            tokenizer,
            coherence_settings,
            code_revision,
            train["model_id"],
            train["model_revision"],
            {row["id"]: row["sha256"] for row in coherence_manifest},
            "pending_blind_human_coherence_0_to_2",
        )
    capability_path = output_dir / "gsm8k.jsonl"
    write_jsonl(capability_path, capability_rows)
    artifacts = {"gsm8k_sha256": sha256_file(capability_path)}
    if not args.capability_only:
        coherence_path = output_dir / "coherence.jsonl"
        blind_path = output_dir / "coherence_blinded.jsonl"
        key_path = output_dir / "coherence_blind_key.json"
        write_jsonl(coherence_path, coherence_rows)
        export_blind(
            coherence_rows,
            args.stage,
            blind_path,
            key_path,
            int(config["blind_review"]["shuffle_seed_base"]) + args.stage,
        )
        artifacts.update(
            {
                "coherence_sha256": sha256_file(coherence_path),
                "blind_packet_sha256": sha256_file(blind_path),
                "blind_key_sha256": sha256_file(key_path),
            }
        )
    summary = {
        "schema_version": 1,
        "status": (
            "capability_regeneration_complete"
            if args.capability_only
            else "technical_complete_pending_blind_coherence_scores"
        ),
        "stage": args.stage,
        "k_values": expected_k,
        "slurm_job_id": slurm_job_id(),
        "code_revision": code_revision,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_matches_metadata": True,
        "strict_load_missing": list(load_result.missing_keys),
        "strict_load_unexpected": list(load_result.unexpected_keys),
        "config_sha256": sha256_file(config_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "train_config_sha256": sha256_file(train_path),
        "capability": capability,
        "coherence_technical": (
            None
            if args.capability_only
            else summarize_generation_rows(
                coherence_rows, int(config["coherence"]["max_new_tokens"])
            )
        ),
        "artifacts": artifacts,
        "caps": {
            "gsm8k_coconut_answer": config["capability"]["max_new_tokens"],
            "coherence_explicit_thinking": config["coherence"]["max_new_tokens"],
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "interpretation": config["interpretation"],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

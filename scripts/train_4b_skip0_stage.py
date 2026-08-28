#!/usr/bin/env python3
"""Train one resumable stage of either frozen matched 4B branch."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import HfApi, model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.batching import pad_causal_records, pad_coconut_records
from mats_latent_safety.constants import optimizer_updates
from mats_latent_safety.coconut import StandardCoconut, initialize_latent_embeddings
from mats_latent_safety.data import clean_gsm8k_row
from mats_latent_safety.fallback import (
    claim_stage_directory,
    validate_authorization,
    validate_matched_batching,
    validate_resume,
)
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    build_training_record,
    ensure_latent_tokens,
    tokenize_reasoning_example,
)
from mats_latent_safety.training import (
    aggregate_token_weighted_loss,
    optimizer_state_bytes,
    percentile,
    run_update,
)


def load_model(config: dict, tokenizer, markers, device: torch.device) -> StandardCoconut:
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        low_cpu_mem_usage=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    anchor_ids = tokenizer.encode("<<", add_special_tokens=False)
    if not anchor_ids:
        raise ValueError("tokenizer produced no anchor token for <<")
    initialize_latent_embeddings(base, markers, anchor_ids[0])
    return StandardCoconut(
        base,
        latent_token_id=markers["latent"],
        start_latent_id=markers["start"],
        end_latent_id=markers["end"],
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)


def make_grouped_batches(
    records,
    groups,
    *,
    latent_token_id: int,
    pad_token_id: int,
    device: torch.device,
    explicit_cot: bool,
):
    batches = []
    for group in groups:
        selected = [records[index] for index in group]
        if explicit_cot:
            batch = pad_causal_records(
                selected,
                pad_token_id=pad_token_id,
                device=device,
            )
        else:
            batch = pad_coconut_records(
                selected,
                latent_token_id=latent_token_id,
                pad_token_id=pad_token_id,
                device=device,
            )
        batches.append(batch)
    return batches


def upload_durable_stage(output_dir: Path, metadata: dict, config: dict) -> dict:
    """Persist the unique stage model while leaving regenerable optimizer state on scratch."""
    repo_id = config["model_checkpoint_durability_repo"]
    path_in_repo = (
        f"{config['durability_path_prefix']}/stage{metadata['completed_stage']}"
    )
    api = HfApi()
    upload_started = time.perf_counter()
    model_commit = api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        allow_patterns=["model_state.pt", "tokenizer/*"],
        commit_message=f"Persist {metadata['branch']} stage {metadata['completed_stage']}",
    )
    model_and_tokenizer_upload_seconds = time.perf_counter() - upload_started
    durability = {
        "status": "model_and_tokenizer_uploaded",
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "model_commit_oid": model_commit.oid,
        "model_commit_url": str(model_commit.commit_url),
        "model_and_tokenizer_upload_seconds": model_and_tokenizer_upload_seconds,
        "metadata_upload_required_for_stage_success": True,
        "optimizer_state_uploaded": False,
        "optimizer_regeneration": "replay_frozen_training_from_previous_durable_stage",
    }
    metadata["durability"] = durability
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    api.upload_file(
        path_or_fileobj=metadata_path,
        path_in_repo=f"{path_in_repo}/metadata.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Record {metadata['branch']} stage {metadata['completed_stage']} metadata",
    )
    return durability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fallback_4b_skip0.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--gsm-train", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume-dir")
    parser.add_argument("--acknowledge-fallback-trigger", default="")
    parser.add_argument("--acknowledge-inline-gate", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    batching_policy = yaml.safe_load(Path(config["matched_batching_policy"]).read_text())
    validate_matched_batching(config, batching_policy)
    if args.stage not in config["stages"]:
        raise ValueError(f"stage {args.stage} is not registered")
    validate_authorization(
        config,
        args.stage,
        args.acknowledge_fallback_trigger,
        args.acknowledge_inline_gate,
    )
    if any(config["forbidden_changes"].values()):
        raise ValueError("a forbidden method change is enabled")
    if config["micro_batch_size"] * config["gradient_accumulation_steps"] != config[
        "effective_batch_size"
    ]:
        raise ValueError("micro-batch and accumulation do not match effective batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if model_info(config["model_id"]).sha != config["model_revision"]:
        raise RuntimeError("resolved base-model revision changed")

    output_dir = claim_stage_directory(Path(args.output_root), args.stage)
    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    code_revision = git_revision()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    source_path = Path(args.gsm_train)
    source = pd.read_parquet(source_path)
    if len(source) != config["dataset_examples"]:
        raise ValueError(f"expected {config['dataset_examples']} rows, found {len(source)}")
    records = []
    for _, row in source.iterrows():
        example = clean_gsm8k_row(str(row.question), str(row.answer))
        tokenized = tokenize_reasoning_example(tokenizer, example)
        records.append(
            build_training_record(
                tokenized,
                stage=args.stage,
                marker_ids=markers,
                c_thought=config["c_thought"],
                max_stage=config["max_latent_stage"],
                explicit_cot=config["branch"] == "explicit_cot",
            )
        )
    lengths = [len(record["input_ids"]) for record in records]
    if max(lengths) > config["max_sequence_length"]:
        raise ValueError("a training record exceeds the frozen maximum sequence length")

    model = load_model(config, tokenizer, markers, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    previous_updates = 0
    if args.stage == 1:
        if args.resume_dir is not None:
            raise ValueError("stage 1 must begin from exact M0, without a resume directory")
    else:
        if args.resume_dir is None:
            raise ValueError("later stages require --resume-dir")
        resume = Path(args.resume_dir)
        metadata = json.loads((resume / "metadata.json").read_text())
        validate_resume(metadata, config, args.stage)
        load_result = model.load_state_dict(
            torch.load(resume / "model_state.pt", map_location=device, weights_only=True),
            strict=True,
        )
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError("strict resume unexpectedly reported key differences")
        optimizer_state = torch.load(
            resume / "optimizer_state.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(optimizer_state)
        previous_updates = int(metadata["cumulative_optimizer_updates"])

    accumulation = config["gradient_accumulation_steps"]
    micro_batch = config["micro_batch_size"]
    update_results = []
    order_hashes = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stage_updates = 0
    for epoch in range(config["epochs_per_stage"]):
        order = list(range(len(records)))
        random.Random(config["seed"] + args.stage * 1000 + epoch).shuffle(order)
        order_hashes.append(sha256_json(order))
        microbatches = [
            order[offset : offset + micro_batch]
            for offset in range(0, len(order), micro_batch)
        ]
        for offset in range(0, len(microbatches), accumulation):
            groups = microbatches[offset : offset + accumulation]
            result = run_update(
                model,
                optimizer,
                make_grouped_batches(
                    records,
                    groups,
                    latent_token_id=markers["latent"],
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    explicit_cot=config["branch"] == "explicit_cot",
                ),
                accumulation,
                check_gradients=(stage_updates + 1) % 50 == 0,
            )
            update_results.append(result)
            stage_updates += 1
            if stage_updates % 10 == 0:
                print(
                    json.dumps(
                        {
                            "stage": args.stage,
                            "epoch": epoch,
                            "stage_update": stage_updates,
                            "seconds": result["seconds"],
                            "loss": result["token_weighted_loss"],
                        }
                    ),
                    flush=True,
                )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    expected_updates = optimizer_updates(
        config["dataset_examples"], config["epochs_per_stage"], config["effective_batch_size"]
    )
    if stage_updates != expected_updates:
        raise RuntimeError(f"observed {stage_updates} updates, expected {expected_updates}")

    model_path = output_dir / "model_state.pt"
    optimizer_path = output_dir / "optimizer_state.pt"
    model_save_started = time.perf_counter()
    torch.save(model.state_dict(), model_path)
    model_save_seconds = time.perf_counter() - model_save_started
    optimizer_save_started = time.perf_counter()
    torch.save(optimizer.state_dict(), optimizer_path)
    optimizer_save_seconds = time.perf_counter() - optimizer_save_started
    tokenizer.save_pretrained(output_dir / "tokenizer")
    hash_started = time.perf_counter()
    model_state_sha256 = sha256_file(model_path)
    optimizer_state_sha256 = sha256_file(optimizer_path)
    checkpoint_hash_seconds = time.perf_counter() - hash_started
    cumulative_updates = previous_updates + stage_updates
    seconds = [float(row["seconds"]) for row in update_results]
    metadata = {
        "schema_version": 1,
        "status": (
            "stage_complete_pending_inline_gate"
            if config["branch"] == "coconut_skip0" and args.stage == 1
            else "stage_complete"
        ),
        "label": config["label"],
        "branch": config["branch"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
        "config_sha256": sha256_json(config),
        "data_sha256": sha256_file(source_path),
        "completed_stage": args.stage,
        "k": 0 if config["branch"] == "explicit_cot" else args.stage * config["c_thought"],
        "epochs": config["epochs_per_stage"],
        "stage_optimizer_updates": stage_updates,
        "cumulative_optimizer_updates": cumulative_updates,
        "order_sha256_by_epoch": order_hashes,
        "training_seconds": training_seconds,
        "mean_update_seconds": statistics.mean(seconds),
        "median_update_seconds": statistics.median(seconds),
        "p95_update_seconds": percentile(seconds, 95),
        "mean_loss": aggregate_token_weighted_loss(update_results),
        "loss_normalization": update_results[0]["loss_normalization"],
        "examples": sum(int(row["examples"]) for row in update_results),
        "nonpadding_tokens": sum(int(row["nonpadding_tokens"]) for row in update_results),
        "supervised_tokens": sum(int(row["supervised_tokens"]) for row in update_results),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "model_state": str(model_path),
        "model_state_sha256": model_state_sha256,
        "optimizer_state": str(optimizer_path),
        "optimizer_state_sha256": optimizer_state_sha256,
        "checkpoint_io": {
            "model_save_seconds": model_save_seconds,
            "optimizer_save_seconds": optimizer_save_seconds,
            "checkpoint_hash_seconds": checkpoint_hash_seconds,
        },
        "gate_required_before_next_stage": args.stage == config.get("gate_after_stage"),
        "training_authorization_basis": config["submission_status"],
        "matched_training_authorized": config["submission_status"]
        in {"inline_gate_passed", "time_pressure_early_cot_stage1_authorized"},
    }
    upload_durable_stage(output_dir, metadata, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

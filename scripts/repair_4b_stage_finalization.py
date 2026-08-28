#!/usr/bin/env python3
"""Finalize and durably upload a completed stage whose metadata step failed."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml

from mats_latent_safety.constants import optimizer_updates
from mats_latent_safety.hashing import sha256_file, sha256_json
from train_4b_skip0_stage import upload_durable_stage


def parse_logged_updates(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith('{"stage"'):
            continue
        row = json.loads(line)
        if "stage_update" in row and "loss" in row:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--gsm-train", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--training-code-revision", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_dir = Path(args.checkpoint_dir)
    model_path = checkpoint_dir / "model_state.pt"
    optimizer_path = checkpoint_dir / "optimizer_state.pt"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    for required in (model_path, optimizer_path, tokenizer_dir):
        if not required.exists():
            raise FileNotFoundError(f"completed-stage artifact is missing: {required}")
    expected_updates = optimizer_updates(
        config["dataset_examples"], config["epochs_per_stage"], config["effective_batch_size"]
    )
    logged = parse_logged_updates(Path(args.training_log))
    if not logged or logged[-1]["stage_update"] != 460 or expected_updates != 468:
        raise RuntimeError(
            "repair is registered only for the two stage-1 jobs that logged through "
            "update 460 and then saved after completing 468 updates"
        )

    order_hashes = []
    for epoch in range(config["epochs_per_stage"]):
        order = list(range(config["dataset_examples"]))
        random.Random(config["seed"] + args.stage * 1000 + epoch).shuffle(order)
        order_hashes.append(sha256_json(order))
    hash_started = time.perf_counter()
    model_hash = sha256_file(model_path)
    optimizer_hash = sha256_file(optimizer_path)
    hash_seconds = time.perf_counter() - hash_started
    snapshot_seconds = [float(row["seconds"]) for row in logged]
    snapshot_losses = [float(row["loss"]) for row in logged]
    metadata = {
        "schema_version": 1,
        "status": (
            "stage_complete_pending_inline_gate"
            if config["branch"] == "coconut_skip0"
            else "stage_complete"
        ),
        "label": config["label"],
        "branch": config["branch"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "code_revision": args.training_code_revision,
        "slurm_job_id": args.training_job_id,
        "config_sha256": sha256_json(config),
        "data_sha256": sha256_file(args.gsm_train),
        "completed_stage": args.stage,
        "k": 0 if config["branch"] == "explicit_cot" else args.stage * config["c_thought"],
        "epochs": config["epochs_per_stage"],
        "stage_optimizer_updates": expected_updates,
        "cumulative_optimizer_updates": expected_updates,
        "order_sha256_by_epoch": order_hashes,
        "training_seconds": None,
        "mean_update_seconds": None,
        "median_update_seconds": None,
        "p95_update_seconds": None,
        "mean_loss": None,
        "loss_normalization": config["loss_normalization"],
        "examples": config["dataset_examples"] * config["epochs_per_stage"],
        "nonpadding_tokens": None,
        "supervised_tokens": None,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "optimizer_state_bytes": None,
        "model_state": str(model_path),
        "model_state_sha256": model_hash,
        "optimizer_state": str(optimizer_path),
        "optimizer_state_sha256": optimizer_hash,
        "checkpoint_io": {
            "model_save_seconds": None,
            "optimizer_save_seconds": None,
            "checkpoint_hash_seconds_repair": hash_seconds,
        },
        "gate_required_before_next_stage": args.stage == config.get("gate_after_stage"),
        "training_authorization_basis": config["submission_status"],
        "matched_training_authorized": config["submission_status"]
        in {"inline_gate_passed", "time_pressure_early_cot_stage1_authorized"},
        "post_training_repair": {
            "reason": "yaml_date_was_not_json_serializable_during_original_metadata_construction",
            "weights_changed": False,
            "completed_updates_evidence": "log_through_460_followed_by_successful_model_and_optimizer_saves_after_loop",
            "unavailable_full_run_metrics": [
                "training_seconds",
                "mean_update_seconds",
                "median_update_seconds",
                "p95_update_seconds",
                "mean_loss",
                "nonpadding_tokens",
                "supervised_tokens",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
                "optimizer_state_bytes",
                "original_checkpoint_save_seconds",
            ],
            "logged_decadal_snapshots": len(logged),
            "logged_decadal_mean_seconds": sum(snapshot_seconds) / len(snapshot_seconds),
            "logged_decadal_mean_loss": sum(snapshot_losses) / len(snapshot_losses),
            "training_log": args.training_log,
            "training_log_sha256": sha256_file(args.training_log),
        },
        "durability": {"status": "pending_repair_upload"},
    }
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    upload_durable_stage(checkpoint_dir, metadata, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

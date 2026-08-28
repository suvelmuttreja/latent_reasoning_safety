#!/usr/bin/env python3
"""Measure access-independent 4B explicit-CoT training memory and throughput."""

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
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.batching import pad_causal_records
from mats_latent_safety.constants import optimizer_updates
from mats_latent_safety.data import clean_gsm8k_row
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import build_training_record, tokenize_reasoning_example
from mats_latent_safety.training import (
    aggregate_token_weighted_loss,
    optimizer_state_bytes,
    percentile,
    run_update,
)


def make_batches(
    records: list[dict[str, list[int] | int]],
    indices: list[int],
    *,
    micro_batch_size: int,
    pad_token_id: int,
    device: torch.device,
    pad_to_length: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    if len(indices) % micro_batch_size:
        raise ValueError("indices must divide evenly into micro-batches")
    return [
        pad_causal_records(
            [records[index] for index in indices[offset : offset + micro_batch_size]],
            pad_token_id=pad_token_id,
            device=device,
            pad_to_length=pad_to_length,
        )
        for offset in range(0, len(indices), micro_batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preflight_4b_cot.yaml")
    parser.add_argument("--gsm-train", required=True)
    parser.add_argument("--calibration-generations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if any(config["forbidden_changes"].values()):
        raise ValueError("a forbidden preflight change is enabled")
    if config["micro_batch_size"] * config["gradient_accumulation_steps"] != config[
        "effective_batch_size"
    ]:
        raise ValueError("micro-batch and accumulation do not match effective batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    code_revision = git_revision()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda")
    revision = config["model_revision"]
    resolved_revision = model_info(config["model_id"]).sha
    if resolved_revision != revision:
        raise RuntimeError(f"resolved revision {resolved_revision} != pinned {revision}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
                stage=0,
                marker_ids={"start": 0, "end": 1, "latent": 2},
                explicit_cot=True,
            )
        )
    lengths = [len(record["input_ids"]) for record in records]
    if max(lengths) > config["max_sequence_length"]:
        raise ValueError(
            f"observed length {max(lengths)} exceeds configured {config['max_sequence_length']}"
        )
    length_summary = {
        "minimum": min(lengths),
        "median": percentile(lengths, 50),
        "p95": percentile(lengths, 95),
        "p99": percentile(lengths, 99),
        "maximum": max(lengths),
        "over_512": sum(length > 512 for length in lengths),
    }

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    accumulation = config["gradient_accumulation_steps"]
    micro_batch = config["memory_micro_batch_size"]
    worst_indices = sorted(range(len(records)), key=lambda i: lengths[i], reverse=True)[:micro_batch]
    worst_indices *= accumulation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_result = run_update(
        model,
        optimizer,
        make_batches(
            records,
            worst_indices,
            micro_batch_size=micro_batch,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            pad_to_length=config["max_sequence_length"],
        ),
        accumulation,
        check_gradients=True,
    )
    memory_result.update(
        {
            "sequence_length": config["max_sequence_length"],
            "source_maximum_sequence_length": max(lengths),
            "source_indices": sorted(set(worst_indices)),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        }
    )

    required_examples = (
        config["warmup_optimizer_updates"] + config["timed_optimizer_updates"]
    ) * config["effective_batch_size"]
    order = list(range(len(records)))
    random.Random(config["seed"]).shuffle(order)
    selected = order[:required_examples]
    cursor = 0
    for _ in range(config["warmup_optimizer_updates"]):
        update_indices = selected[cursor : cursor + config["effective_batch_size"]]
        cursor += config["effective_batch_size"]
        run_update(
            model,
            optimizer,
            make_batches(
                records,
                update_indices,
                micro_batch_size=config["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            ),
            accumulation,
            check_gradients=False,
        )

    torch.cuda.reset_peak_memory_stats()
    timed = []
    total_started = time.perf_counter()
    for update in range(config["timed_optimizer_updates"]):
        update_indices = selected[cursor : cursor + config["effective_batch_size"]]
        cursor += config["effective_batch_size"]
        timed.append(
            run_update(
                model,
                optimizer,
                make_batches(
                    records,
                    update_indices,
                    micro_batch_size=config["micro_batch_size"],
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                ),
                accumulation,
                check_gradients=update + 1 == config["timed_optimizer_updates"],
            )
        )
        print(
            json.dumps(
                {
                    "update": update + 1,
                    "seconds": timed[-1]["seconds"],
                    "loss": timed[-1]["token_weighted_loss"],
                }
            ),
            flush=True,
        )
    total_seconds = time.perf_counter() - total_started
    update_seconds = [float(row["seconds"]) for row in timed]
    total_examples = sum(int(row["examples"]) for row in timed)
    total_nonpadding = sum(int(row["nonpadding_tokens"]) for row in timed)
    total_supervised = sum(int(row["supervised_tokens"]) for row in timed)
    full_updates_per_stage = optimizer_updates(
        config["dataset_examples"],
        config["epochs_per_coconut_stage"],
        config["effective_batch_size"],
    )
    full_updates = full_updates_per_stage * config["coconut_stages"]
    mean_update_seconds = statistics.mean(update_seconds)
    projected_cot_seconds = mean_update_seconds * full_updates

    calibration_projection = None
    if args.calibration_generations:
        rows = [
            json.loads(line)
            for line in Path(args.calibration_generations).read_text().splitlines()
            if line.strip()
        ]
        if rows:
            mean_generated = statistics.mean(row["generated_tokens"] for row in rows)
            aggregate_tps = sum(row["generated_tokens"] for row in rows) / sum(
                row["seconds"] for row in rows
            )
            dense_prompts = 200 + 60 + 12 + 10
            seconds_per_explicit_checkpoint = dense_prompts * mean_generated / aggregate_tps
            calibration_projection = {
                "calibration_examples": len(rows),
                "mean_generated_tokens": mean_generated,
                "aggregate_tokens_per_second": aggregate_tps,
                "dense_prompts_per_checkpoint": dense_prompts,
                "projected_seconds_per_explicit_checkpoint": seconds_per_explicit_checkpoint,
                "scope": "rough M0-derived native-generation projection; not Coconut timing",
            }

    result = {
        "schema_version": 1,
        "status": "passed",
        "label": config["label"],
        "model_id": config["model_id"],
        "model_revision": revision,
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
        "config_sha256": sha256_json(config),
        "data_sha256": sha256_file(source_path),
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "sequence_lengths": length_summary,
        "memory_preflight": memory_result,
        "throughput": {
            "warmup_optimizer_updates": config["warmup_optimizer_updates"],
            "timed_optimizer_updates": config["timed_optimizer_updates"],
            "examples": total_examples,
            "nonpadding_tokens": total_nonpadding,
            "supervised_tokens": total_supervised,
            "total_seconds": total_seconds,
            "mean_update_seconds": mean_update_seconds,
            "median_update_seconds": statistics.median(update_seconds),
            "p95_update_seconds": percentile(update_seconds, 95),
            "examples_per_second": total_examples / total_seconds,
            "nonpadding_tokens_per_second": total_nonpadding / total_seconds,
            "supervised_tokens_per_second": total_supervised / total_seconds,
            "mean_loss": aggregate_token_weighted_loss(timed),
            "loss_normalization": timed[0]["loss_normalization"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "projection": {
            "optimizer_updates_per_stage": full_updates_per_stage,
            "total_cot_optimizer_updates": full_updates,
            "projected_cot_training_seconds": projected_cot_seconds,
            "projected_cot_gpu_hours": projected_cot_seconds / 3600,
            "model_parameter_bytes": model_parameter_bytes,
            "model_only_stage_checkpoints": config["coconut_stages"],
            "projected_cot_model_only_checkpoint_bytes": model_parameter_bytes
            * config["coconut_stages"],
            "projected_six_matched_model_only_checkpoint_bytes": model_parameter_bytes * 6,
        },
        "native_generation_projection": calibration_projection,
        "scope": {
            "cot_memory_and_throughput": "measured",
            "coconut_stage1_throughput": "pending_public_checkpoint_key_audit",
            "matched_training_authorized": False,
            "checkpoint_saved": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

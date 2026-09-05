#!/usr/bin/env python3
"""Small self-contained skip0 validation while public 4B approval is pending.

This is an infrastructure/method validation, never a scientific safety result.
It exercises all three registered skip0 stages and a fixed-weight K=0/K=6
capability check on frozen held-out examples.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.batching import pad_coconut_records
from mats_latent_safety.coconut import StandardCoconut, initialize_latent_embeddings
from mats_latent_safety.data import clean_gsm8k_row
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.parsing import extract_gsm8k_answer
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    build_coconut_question,
    build_training_record,
    ensure_latent_tokens,
    tokenize_reasoning_example,
)


def load_model(config: dict, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
    )
    base.resize_token_embeddings(len(tokenizer))
    anchor = tokenizer.encode("<<", add_special_tokens=False)[0]
    initialize_latent_embeddings(base, markers, anchor)
    model = StandardCoconut(
        base,
        latent_token_id=markers["latent"],
        start_latent_id=markers["start"],
        end_latent_id=markers["end"],
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)
    return model, tokenizer, markers


def gradients_finite(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients)


def train_stage(model, tokenizer, markers, examples, stage: int, config: dict, optimizer):
    accumulation = config["gradient_accumulation_steps"]
    micro_batch_size = config["micro_batch_size"]
    losses, supervised_tokens, updates = [], 0, 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter()
    examples_processed = 0
    for epoch in range(config["epochs_per_stage"]):
        order = list(range(len(examples)))
        random.Random(config["seed"] + stage * 1000 + epoch).shuffle(order)
        microbatches = [
            order[offset : offset + micro_batch_size]
            for offset in range(0, len(order), micro_batch_size)
        ]
        for offset, example_indices in enumerate(microbatches):
            records = []
            for example_index in example_indices:
                tokenized = tokenize_reasoning_example(tokenizer, examples[example_index])
                records.append(
                    build_training_record(
                        tokenized,
                        stage=stage,
                        marker_ids=markers,
                        c_thought=config["c_thought"],
                        max_stage=3,
                    )
                )
            device = next(model.parameters()).device
            batch = pad_coconut_records(
                records,
                latent_token_id=markers["latent"],
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            output = model(**batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite loss at stage {stage}, examples {example_indices}")
            (output.loss / accumulation).backward()
            losses.append(float(output.loss.detach().float().cpu()))
            supervised_tokens += sum(int(record["supervised_tokens"]) for record in records)
            examples_processed += len(records)
            final_microbatch = offset + 1 == len(microbatches)
            if (offset + 1) % accumulation == 0 or final_microbatch:
                if not gradients_finite(model):
                    raise RuntimeError(
                        f"non-finite gradients at stage {stage}, examples {example_indices}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
    torch.cuda.synchronize()
    return {
        "stage": stage,
        "k": stage * config["c_thought"],
        "unique_examples": len(examples),
        "examples": examples_processed,
        "epochs": config["epochs_per_stage"],
        "optimizer_updates": updates,
        "supervised_tokens": supervised_tokens,
        "mean_loss": sum(losses) / len(losses),
        "seconds": time.perf_counter() - started,
    }


def evaluate(model, tokenizer, markers, records, k: int, max_new_tokens: int):
    model.eval()
    outputs = []
    correct = 0
    started = time.perf_counter()
    for row in records:
        example = {"question": row["question"], "steps": ["placeholder"], "answer": row["answer"]}
        tokenized = tokenize_reasoning_example(tokenizer, example)
        scaffold = build_coconut_question(tokenized.question, markers, k)
        device = next(model.parameters()).device
        generated = model.generate_from_scaffold(
            torch.tensor([scaffold["input_ids"]], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            seed=42,
        )
        decoded = tokenizer.decode(generated.token_ids, skip_special_tokens=True)
        predicted = extract_gsm8k_answer(decoded)
        reference = extract_gsm8k_answer(str(row["answer"]))
        is_correct = predicted == reference
        correct += int(is_correct)
        outputs.append(
            {
                "prompt_id": row["id"],
                "k": k,
                "raw_output": decoded,
                "predicted_answer": predicted,
                "reference_answer": reference,
                "correct": is_correct,
                "generated_tokens": generated.generated_tokens,
                "stop_reason": generated.stop_reason,
            }
        )
    torch.cuda.synchronize()
    return {
        "k": k,
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "seconds": time.perf_counter() - started,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation_0_6b.yaml")
    parser.add_argument("--gsm-train", required=True)
    parser.add_argument("--eval-manifest", default="manifests/gsm8k_calibration_20.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    code_revision = git_revision()
    config = yaml.safe_load(Path(args.config).read_text())
    if any(config["forbidden_changes"].values()):
        raise ValueError("a forbidden method change is enabled")
    if (
        config["micro_batch_size"] * config["gradient_accumulation_steps"]
        != config["effective_batch_size"]
    ):
        raise ValueError("micro-batch and accumulation do not match effective batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model, tokenizer, markers = load_model(config, device)
    source = pd.read_parquet(args.gsm_train).iloc[: config["train_examples"]]
    examples = [clean_gsm8k_row(str(row.question), str(row.answer)) for _, row in source.iterrows()]
    eval_rows = json.loads(Path(args.eval_manifest).read_text())["records"][
        : config["eval_examples"]
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_results = []
    for stage in config["stages"]:
        stage_result = train_stage(model, tokenizer, markers, examples, stage, config, optimizer)
        checkpoint = output_dir / f"checkpoint_stage{stage}.pt"
        torch.save(model.state_dict(), checkpoint)
        stage_result["checkpoint"] = str(checkpoint)
        stage_result["checkpoint_sha256"] = sha256_file(checkpoint)
        stage_results.append(stage_result)

    evaluations = {
        "k0": evaluate(
            model, tokenizer, markers, eval_rows, 0, config["generation_max_new_tokens"]
        ),
        "k6": evaluate(
            model, tokenizer, markers, eval_rows, 6, config["generation_max_new_tokens"]
        ),
    }
    final_checkpoint = output_dir / "checkpoint_stage3.pt"
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, tokenizer2, markers2 = load_model(config, device)
    load_result = reloaded.load_state_dict(
        torch.load(final_checkpoint, map_location=device, weights_only=True), strict=True
    )
    reload_probe = evaluate(reloaded, tokenizer2, markers2, eval_rows[:1], 6, 32)
    result = {
        "schema_version": 1,
        "status": "passed",
        "label": config["label"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
        "config_sha256": sha256_json(config),
        "training_source_indices": list(range(config["train_examples"])),
        "stage_results": stage_results,
        "evaluations": evaluations,
        "strict_reload_missing": list(load_result.missing_keys),
        "strict_reload_unexpected": list(load_result.unexpected_keys),
        "reload_probe": reload_probe,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "assessment": {
            "both_k_execute": True,
            "k_dependence_accuracy_points": evaluations["k6"]["accuracy"]
            - evaluations["k0"]["accuracy"],
            "matched_4b_training_authorized": False,
            "reason": (
                "the 4B memory/throughput preflights and either public or fallback "
                "in-line Gate -1 must pass first"
            ),
        },
    }
    (output_dir / "validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

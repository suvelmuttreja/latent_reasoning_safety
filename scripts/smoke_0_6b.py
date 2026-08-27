#!/usr/bin/env python3
"""One-step Qwen3-0.6B standard-Coconut save/reload/generation smoke."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.coconut import StandardCoconut, initialize_latent_embeddings
from mats_latent_safety.data import clean_gsm8k_row
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    build_coconut_question,
    build_training_record,
    ensure_latent_tokens,
    tokenize_reasoning_example,
)


def finite_gradients(model: torch.nn.Module) -> tuple[bool, float]:
    total = torch.zeros((), device=next(model.parameters()).device)
    seen = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                return False, math.nan
            total += parameter.grad.detach().float().pow(2).sum()
            seen += parameter.grad.numel()
    return seen > 0, float(total.sqrt().cpu())


def load_wrapped(config: dict, revision: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    marker_ids = ensure_latent_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    anchor_ids = tokenizer.encode("<<", add_special_tokens=False)
    if not anchor_ids:
        raise ValueError("tokenizer produced no anchor token for <<")
    initialize_latent_embeddings(model, marker_ids, anchor_ids[0])
    wrapper = StandardCoconut(
        model,
        latent_token_id=marker_ids["latent"],
        start_latent_id=marker_ids["start"],
        end_latent_id=marker_ids["end"],
        eos_token_id=tokenizer.eos_token_id,
    ).to(device)
    return wrapper, tokenizer, marker_ids


def generate(wrapper, tokenizer, marker_ids, tokenized, k: int) -> dict:
    prompt = build_coconut_question(tokenized.question, marker_ids, k)
    device = next(wrapper.parameters()).device
    result = wrapper.generate_from_scaffold(
        torch.tensor([prompt["input_ids"]], device=device),
        max_new_tokens=64,
        do_sample=False,
        seed=42,
    )
    return {
        "k": k,
        "token_ids": result.token_ids,
        "decoded": tokenizer.decode(result.token_ids, skip_special_tokens=False),
        "stop_reason": result.stop_reason,
        "generated_tokens": result.generated_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_0_6b.yaml")
    parser.add_argument("--gsm-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    code_revision = git_revision()
    if not torch.cuda.is_available():
        raise RuntimeError("the registered smoke requires a CUDA GPU")
    config = yaml.safe_load(Path(args.config).read_text())
    forbidden = [name for name, enabled in config["forbidden_changes"].items() if enabled]
    if forbidden:
        raise ValueError(f"forbidden method changes enabled: {forbidden}")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    resolved_revision = model_info(config["model_id"]).sha
    revision = config["model_revision"]
    if resolved_revision != revision:
        raise RuntimeError(
            f"pinned model revision {revision} no longer matches resolved main {resolved_revision}"
        )
    wrapper, tokenizer, marker_ids = load_wrapped(config, revision, device)

    rows = pd.read_parquet(args.gsm_parquet)
    example = clean_gsm8k_row(str(rows.iloc[0].question), str(rows.iloc[0].answer))
    tokenized = tokenize_reasoning_example(tokenizer, example)
    record = build_training_record(
        tokenized,
        stage=config["stage"],
        marker_ids=marker_ids,
        c_thought=config["c_thought"],
        max_stage=3,
    )
    if len(record["input_ids"]) > config["max_sequence_length"]:
        raise ValueError("smoke example exceeds configured sequence length")
    batch = {
        key: torch.tensor([record[key]], device=device)
        for key in ("input_ids", "attention_mask", "labels", "position_ids")
    }
    wrapper.train()
    optimizer = torch.optim.AdamW(
        wrapper.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    start = time.perf_counter()
    output = wrapper(**batch)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite smoke loss: {output.loss}")
    output.loss.backward()
    grads_finite, grad_norm = finite_gradients(wrapper)
    if not grads_finite:
        raise RuntimeError("missing or non-finite gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start
    loss_value = float(output.loss.detach().float().cpu())
    wrapper.eval()
    before_reload = {str(k): generate(wrapper, tokenizer, marker_ids, tokenized, k) for k in (0, 2)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint_stage1_step1.pt"
    torch.save(wrapper.state_dict(), checkpoint)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    del optimizer, output, wrapper
    gc.collect()
    torch.cuda.empty_cache()

    reloaded, tokenizer2, marker_ids2 = load_wrapped(config, revision, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    load_result = reloaded.load_state_dict(state, strict=True)
    reloaded.eval()
    after_reload = {
        str(k): generate(reloaded, tokenizer2, marker_ids2, tokenized, k) for k in (0, 2)
    }
    deterministic = all(
        before_reload[key]["token_ids"] == after_reload[key]["token_ids"]
        for key in before_reload
    )
    if not deterministic:
        raise RuntimeError("save/reload changed greedy latent generation")
    summary = {
        "status": "passed",
        "model_id": config["model_id"],
        "model_revision": revision,
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
        "method": config["method"],
        "loss": loss_value,
        "training_step_seconds": step_seconds,
        "gradients_finite": grads_finite,
        "gradient_norm": grad_norm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "strict_load_missing": list(load_result.missing_keys),
        "strict_load_unexpected": list(load_result.unexpected_keys),
        "save_reload_generation_identical": deterministic,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "processed_example": example,
        "training_record": {
            "sequence_tokens": len(record["input_ids"]),
            "k": record["k"],
            "skipped_steps": record["skipped_steps"],
            "supervised_tokens": record["supervised_tokens"],
        },
        "generation": after_reload,
        "config_sha256": sha256_json(config),
    }
    (output_dir / "smoke_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

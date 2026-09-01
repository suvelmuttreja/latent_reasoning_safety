#!/usr/bin/env python3
"""Evaluate M0 or CoT-u3 on GSM8K-200 with the same native-chat harness."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.parsing import extract_gsm8k_answer, parse_thinking_response
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    build_explicit_question,
    ensure_latent_tokens,
    tokenize_native_chat_prompt,
)
from train_4b_skip0_stage import load_model


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/native_gsm8k_endpoint_controls.yaml")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-root", default=os.environ.get("WORK_ROOT"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("status") != "frozen_before_generation":
        raise ValueError("native GSM8K control config is not frozen")
    if args.condition not in config["conditions"]:
        raise ValueError("unregistered native GSM8K condition")
    condition = config["conditions"][args.condition]
    manifest_path = Path(config["manifest"])
    records = json.loads(manifest_path.read_text())["records"]
    sampling = config["sampling"]
    settings = {
        "serialization": config["serialization"],
        "do_sample": bool(sampling["do_sample"]),
        "max_new_tokens": int(sampling["max_new_tokens"]),
        "stop_tokens": list(sampling["stop_tokens"]),
    }
    generation_config_sha256 = sha256_json(settings)
    implementation_sha256 = sha256_json(
        {
            "evaluator": sha256_file(Path(__file__)),
            "serialization": sha256_file(
                Path("src/mats_latent_safety/serialization.py")
            ),
            "parser": sha256_file(Path("src/mats_latent_safety/parsing.py")),
        }
    )

    train_path = Path(config["train_config"])
    train = yaml.safe_load(train_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        train["model_id"], revision=train["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint_identity = train["model_revision"]
    if condition["model_source"] == "huggingface_base_revision":
        model = AutoModelForCausalLM.from_pretrained(
            condition["model_id"],
            revision=condition["model_revision"],
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        markers = None
    elif condition["model_source"] == "matched_checkpoint":
        if not args.work_root:
            raise ValueError("--work-root or WORK_ROOT is required for checkpoint control")
        markers = ensure_latent_tokens(tokenizer)
        model = load_model(train, tokenizer, markers, torch.device("cuda"))
        checkpoint_dir = Path(args.work_root) / "results" / condition["checkpoint_subdir"]
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
        checkpoint = checkpoint_dir / "model_state.pt"
        checkpoint_identity = sha256_file(checkpoint)
        if metadata["branch"] != "explicit_cot":
            raise ValueError("native CoT control requires an explicit-CoT checkpoint")
        if (
            metadata["completed_stage"] != int(condition["checkpoint_stage"])
            or metadata["k"] != int(condition["checkpoint_k"])
        ):
            raise ValueError("CoT checkpoint stage/K differs from frozen control")
        if checkpoint_identity != condition["checkpoint_sha256"]:
            raise ValueError("CoT checkpoint hash differs from frozen control")
        if metadata["model_state_sha256"] != checkpoint_identity:
            raise ValueError("CoT checkpoint hash differs from metadata")
        if metadata["durability"]["status"] != "model_and_tokenizer_uploaded":
            raise ValueError("CoT checkpoint lacks durable upload evidence")
        state = torch.load(checkpoint, map_location="cuda", weights_only=True)
        load_result = model.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError("strict CoT checkpoint load returned key differences")
    else:
        raise ValueError("unsupported native GSM8K model source")
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "generations.jsonl"
    partial_path = output_dir / "generations.partial.jsonl"
    if final_path.exists():
        raise FileExistsError("refusing to overwrite completed native GSM8K control")
    existing = []
    if partial_path.exists():
        existing = [
            json.loads(line)
            for line in partial_path.read_text().splitlines()
            if line.strip()
        ]
    if [row.get("prompt_id") for row in existing] != [
        row["id"] for row in records[: len(existing)]
    ]:
        raise ValueError("native GSM8K partial cache is not a manifest-order prefix")
    for row in existing:
        if (
            row.get("condition") != args.condition
            or row.get("checkpoint_sha256") != checkpoint_identity
            or row.get("generation_config_sha256") != generation_config_sha256
            or row.get("generation_implementation_sha256") != implementation_sha256
        ):
            raise ValueError("native GSM8K partial cache provenance differs from this run")

    code_revision = git_revision()
    for index, record in enumerate(records[len(existing) :], start=len(existing)):
        prefix_ids, rendered_prefix = tokenize_native_chat_prompt(
            tokenizer, record["question"].strip()
        )
        started = time.perf_counter()
        if markers is None:
            batch = tokenizer(rendered_prefix, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                sequence = model.generate(
                    **batch,
                    max_new_tokens=int(sampling["max_new_tokens"]),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = sequence[0, batch["input_ids"].shape[1] :]
            stop_reason = (
                "eos_token"
                if len(generated_ids)
                and int(generated_ids[-1]) == int(tokenizer.eos_token_id)
                else "length"
            )
        else:
            scaffold = build_explicit_question(prefix_ids)
            generated = model.generate_from_scaffold(
                torch.tensor([scaffold["input_ids"]], device=next(model.parameters()).device),
                max_new_tokens=int(sampling["max_new_tokens"]),
                do_sample=False,
                seed=int(sampling["seed"]) + index,
            )
            generated_ids = generated.token_ids
            stop_reason = generated.stop_reason
        torch.cuda.synchronize()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_thinking_response(decoded)
        prediction = extract_gsm8k_answer(parsed.final_answer)
        reference = extract_gsm8k_answer(record["answer"])
        row = {
            "schema_version": 1,
            "condition": args.condition,
            "prompt_id": record["id"],
            "prompt_sha256": record["sha256"],
            "model_id": train["model_id"],
            "model_revision": train["model_revision"],
            "checkpoint_sha256": checkpoint_identity,
            "code_revision": code_revision,
            "slurm_job_id": slurm_job_id(),
            "k": 0,
            "raw_serialized_input": rendered_prefix,
            "serialized_input_ids_sha256": sha256_json(prefix_ids),
            "raw_output": decoded,
            "parsed_thinking": parsed.thinking,
            "parsed_final_answer": parsed.final_answer,
            "predicted_answer": prediction,
            "reference_answer": reference,
            "correct": prediction == reference,
            "input_tokens": len(prefix_ids),
            "generated_tokens": len(generated_ids),
            "thinking_tokens": len(
                tokenizer.encode(parsed.thinking or "", add_special_tokens=False)
            ),
            "answer_tokens": len(
                tokenizer.encode(parsed.final_answer, add_special_tokens=False)
            ),
            "stop_reason": stop_reason,
            "truncated": stop_reason == "length",
            "seconds": time.perf_counter() - started,
            "generation_config_sha256": generation_config_sha256,
            "generation_implementation_sha256": implementation_sha256,
        }
        append_row(partial_path, row)
        existing.append(row)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "prompt_id": row["prompt_id"],
                    "correct": row["correct"],
                    "generated_tokens": row["generated_tokens"],
                    "stop_reason": row["stop_reason"],
                }
            ),
            flush=True,
        )

    if len(existing) != len(records):
        raise RuntimeError("native GSM8K control cache is incomplete")
    partial_path.replace(final_path)
    correct = sum(bool(row["correct"]) for row in existing)
    truncations = sum(bool(row["truncated"]) for row in existing)
    threshold = float(config["cap_guard"]["threshold"])
    guard_passed = truncations / len(existing) < threshold
    summary = {
        "schema_version": 1,
        "status": "complete" if guard_passed else "complete_cap_guard_failed",
        "condition": args.condition,
        "rows": len(existing),
        "correct": correct,
        "observed_accuracy": correct / len(existing),
        "truncations": truncations,
        "accuracy_bounds_without_imputation": [
            correct / len(existing),
            (correct + truncations) / len(existing),
        ],
        "cap_guard_threshold": threshold,
        "cap_guard_passed": guard_passed,
        "checkpoint_sha256": checkpoint_identity,
        "generation_config_sha256": generation_config_sha256,
        "generation_implementation_sha256": implementation_sha256,
        "generations_sha256": sha256_file(final_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


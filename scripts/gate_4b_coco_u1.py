#!/usr/bin/env python3
"""Evaluate the self-trained 4B stage-1 endpoint for in-line method validity."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.parsing import extract_gsm8k_answer
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import build_coconut_question, ensure_latent_tokens
from train_4b_skip0_stage import load_model


def read_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload["records"]
    if not records:
        raise ValueError(f"manifest has no records: {path}")
    return records


def question_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt.strip() + "\n", add_special_tokens=True)


def generate(model, tokenizer, markers, prompt: str, k: int, settings: dict, seed: int):
    scaffold = build_coconut_question(question_ids(tokenizer, prompt), markers, k)
    device = next(model.parameters()).device
    started = time.perf_counter()
    result = model.generate_from_scaffold(
        torch.tensor([scaffold["input_ids"]], device=device),
        max_new_tokens=settings["max_new_tokens"],
        do_sample=settings["do_sample"],
        temperature=settings.get("temperature", 0.6),
        top_p=settings.get("top_p", 0.95),
        top_k=settings.get("top_k", 20),
        seed=seed,
    )
    decoded = tokenizer.decode(result.token_ids, skip_special_tokens=True).strip()
    return {
        "raw_output": decoded,
        "parsed_final_answer": decoded,
        "generated_tokens": result.generated_tokens,
        "stop_reason": result.stop_reason,
        "truncated": result.stop_reason == "length",
        "seconds": time.perf_counter() - started,
    }


def evaluate_capability(model, tokenizer, markers, records, settings: dict, seed: int):
    by_k = {}
    for k in settings["k_values"]:
        outputs, correct = [], 0
        for index, row in enumerate(records):
            generated = generate(
                model, tokenizer, markers, row["question"], k, settings, seed + index
            )
            prediction = extract_gsm8k_answer(generated["raw_output"])
            reference = extract_gsm8k_answer(row["answer"])
            is_correct = prediction == reference
            correct += int(is_correct)
            outputs.append(
                {
                    "prompt_id": row["id"],
                    "k": k,
                    "predicted_answer": prediction,
                    "reference_answer": reference,
                    "correct": is_correct,
                    **generated,
                }
            )
        by_k[str(k)] = {
            "k": k,
            "examples": len(outputs),
            "correct": correct,
            "accuracy": correct / len(outputs),
            "mean_generated_tokens": statistics.mean(
                row["generated_tokens"] for row in outputs
            ),
            "truncated": sum(int(row["truncated"]) for row in outputs),
            "outputs": outputs,
        }
    low, high = (str(value) for value in settings["k_values"])
    low_rows = by_k[low]["outputs"]
    high_rows = by_k[high]["outputs"]
    return {
        "by_k": by_k,
        "accuracy_delta_high_minus_low": by_k[high]["accuracy"] - by_k[low]["accuracy"],
        "paired_prediction_changed": sum(
            left["predicted_answer"] != right["predicted_answer"]
            for left, right in zip(low_rows, high_rows, strict=True)
        ),
        "paired_output_changed": sum(
            left["raw_output"] != right["raw_output"]
            for left, right in zip(low_rows, high_rows, strict=True)
        ),
    }


def evaluate_coherence(
    model,
    tokenizer,
    markers,
    records,
    settings: dict,
    seed: int,
    model_id: str,
    model_revision: str,
):
    outputs = []
    for k in settings["k_values"]:
        for index, row in enumerate(records):
            generated = generate(
                model, tokenizer, markers, row["prompt"], k, settings, seed + index
            )
            outputs.append(
                {
                    "prompt_id": row["id"],
                    "prompt_kind": row["kind"],
                    "prompt_sha256": row["sha256"],
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "k": k,
                    **generated,
                }
            )
    return outputs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_4b_coco_u1.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gate_config_path = Path(args.config)
    gate_config = yaml.safe_load(gate_config_path.read_text())
    train_config_path = Path(gate_config["train_config"])
    train_config = yaml.safe_load(train_config_path.read_text())
    checkpoint_dir = Path(args.checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    if metadata["completed_stage"] != gate_config["checkpoint_stage"]:
        raise ValueError("checkpoint is not the registered stage")
    if metadata["k"] != gate_config["checkpoint_k"]:
        raise ValueError("checkpoint does not have the registered K")
    if metadata["model_id"] != train_config["model_id"]:
        raise ValueError("checkpoint model ID differs from the frozen train config")
    if metadata["model_revision"] != train_config["model_revision"]:
        raise ValueError("checkpoint model revision differs from the frozen train config")
    if metadata["config_sha256"] != sha256_json(train_config):
        raise ValueError("checkpoint training config hash differs from the frozen config")
    checkpoint_path = checkpoint_dir / "model_state.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != metadata["model_state_sha256"]:
        raise ValueError("checkpoint hash differs from stage metadata")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(gate_config["seed"])
    torch.cuda.manual_seed_all(gate_config["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        train_config["model_id"], revision=train_config["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    model = load_model(train_config, tokenizer, markers, device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()

    capability_manifest = Path(gate_config["capability"]["manifest"])
    coherence_manifest = Path(gate_config["coherence"]["manifest"])
    capability = evaluate_capability(
        model,
        tokenizer,
        markers,
        read_manifest(capability_manifest),
        gate_config["capability"],
        gate_config["seed"],
    )
    coherence = evaluate_coherence(
        model,
        tokenizer,
        markers,
        read_manifest(coherence_manifest),
        gate_config["coherence"],
        gate_config["seed"],
        train_config["model_id"],
        train_config["model_revision"],
    )
    coherence_path = output_dir / "coherence_generations.jsonl"
    write_jsonl(coherence_path, coherence)
    summary = {
        "schema_version": 1,
        "status": "pending_blind_coherence_and_method_review",
        "next_stage_authorized": False,
        "model_id": train_config["model_id"],
        "model_revision": train_config["model_revision"],
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "checkpoint_stage": metadata["completed_stage"],
        "checkpoint_k": metadata["k"],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_matches_stage_metadata": True,
        "strict_load_missing": list(load_result.missing_keys),
        "strict_load_unexpected": list(load_result.unexpected_keys),
        "gate_config_sha256": sha256_json(gate_config),
        "train_config_sha256": sha256_json(train_config),
        "capability_manifest_sha256": sha256_file(capability_manifest),
        "coherence_manifest_sha256": sha256_file(coherence_manifest),
        "capability": capability,
        "coherence_generations": str(coherence_path),
        "coherence_generations_sha256": sha256_file(coherence_path),
        "coherence_outputs": len(coherence),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "decision_inputs_forbidden": gate_config["decision"]["forbidden_inputs"],
    }
    (output_dir / "gate_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"capability"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

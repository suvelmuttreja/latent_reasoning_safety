#!/usr/bin/env python3
"""Generate the frozen full-60 M0 safety baseline without loading a judge."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.parsing import parse_thinking_response
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import serialize_native_chat


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_manifest(path: Path) -> list[dict]:
    records = json.loads(path.read_text())["records"]
    if not records:
        raise ValueError("M0 safety manifest is empty")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_full60_safety.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("status") != "frozen_before_generation":
        raise ValueError("M0 full-60 config is not frozen for generation")
    if config.get("condition") != "m0":
        raise ValueError("M0 full-60 condition must be m0")
    if config.get("serialization") != "native_qwen_chat":
        raise ValueError("M0 full-60 requires the selected native chat format")

    manifest_path = Path(config["manifest"])
    records = read_manifest(manifest_path)
    sampling = config["sampling"]
    generation_settings = {
        "serialization": config["serialization"],
        "do_sample": bool(sampling["do_sample"]),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "top_k": int(sampling["top_k"]),
        "max_new_tokens": int(sampling["max_new_tokens"]),
        "stop_tokens": list(sampling["stop_tokens"]),
        "seed_rule": sampling["seed_rule"],
    }
    generation_config_sha256 = sha256_json(generation_settings)
    implementation_sha256 = sha256_json(
        {
            "generator": sha256_file(Path(__file__)),
            "serialization": sha256_file(
                Path("src/mats_latent_safety/serialization.py")
            ),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "generations.jsonl"
    partial_path = output_dir / "generations.partial.jsonl"
    if final_path.exists():
        raise FileExistsError("refusing to overwrite completed M0 generations")
    existing = []
    if partial_path.exists():
        existing = [
            json.loads(line)
            for line in partial_path.read_text().splitlines()
            if line.strip()
        ]
    expected_ids = [record["id"] for record in records[: len(existing)]]
    if [row.get("prompt_id") for row in existing] != expected_ids:
        raise ValueError("M0 partial cache is not a manifest-order prefix")
    for row in existing:
        if (
            row.get("condition") != "m0"
            or row.get("model_revision") != config["model_revision"]
            or row.get("generation_config_sha256") != generation_config_sha256
            or row.get("generation_implementation_sha256") != implementation_sha256
            or row.get("evaluator_payload") is not None
            or row.get("evaluator_score") is not None
        ):
            raise ValueError("M0 partial cache provenance differs from this run")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    code_revision = git_revision()

    for index, record in enumerate(records[len(existing) :], start=len(existing)):
        prompt = record["forbidden_prompt"]
        serialized = serialize_native_chat(tokenizer, prompt)
        batch = tokenizer(serialized, return_tensors="pt").to(model.device)
        seed = int(sampling["seed"]) + index
        set_seed(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            sequence = model.generate(
                **batch,
                max_new_tokens=int(sampling["max_new_tokens"]),
                do_sample=bool(sampling["do_sample"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generated_ids = sequence[0, batch["input_ids"].shape[1] :]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_thinking_response(decoded)
        hit_eos = bool(
            len(generated_ids)
            and int(generated_ids[-1]) == int(tokenizer.eos_token_id)
        )
        stop_reason = "eos_token" if hit_eos else "length"
        row = {
            "schema_version": 1,
            "generation_purpose": config["purpose"],
            "condition": "m0",
            "prompt_id": record["id"],
            "prompt_sha256": record["sha256"],
            "category": record["category"],
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "checkpoint_sha256": config["model_revision"],
            "checkpoint_identity_kind": "huggingface_model_revision",
            "code_revision": code_revision,
            "slurm_job_id": slurm_job_id(),
            "branch": "m0",
            "k": 0,
            "sampling_seed": seed,
            "generation_config_sha256": generation_config_sha256,
            "generation_implementation_sha256": implementation_sha256,
            "raw_serialized_input": serialized,
            "serialized_input_ids_sha256": sha256_json(
                batch["input_ids"][0].tolist()
            ),
            "raw_output": decoded,
            "parsed_thinking": parsed.thinking,
            "parsed_final_answer": parsed.final_answer,
            "input_tokens": int(batch["input_ids"].shape[1]),
            "generated_tokens": int(len(generated_ids)),
            "thinking_tokens": len(
                tokenizer.encode(parsed.thinking or "", add_special_tokens=False)
            ),
            "answer_tokens": len(
                tokenizer.encode(parsed.final_answer, add_special_tokens=False)
            ),
            "stop_reason": stop_reason,
            "truncated": stop_reason == "length",
            "seconds": time.perf_counter() - started,
            "evaluator_payload": None,
            "evaluator_score": None,
        }
        append_row(partial_path, row)
        existing.append(row)
        print(
            json.dumps(
                {
                    "condition": "m0",
                    "prompt_id": row["prompt_id"],
                    "generated_tokens": row["generated_tokens"],
                    "stop_reason": row["stop_reason"],
                }
            ),
            flush=True,
        )

    if len(existing) != len(records):
        raise RuntimeError("M0 generation cache is incomplete")
    partial_path.replace(final_path)
    truncations = sum(bool(row["truncated"]) for row in existing)
    threshold = float(config["cap_guard"]["threshold"])
    guard_passed = truncations / len(existing) < threshold
    summary = {
        "schema_version": 1,
        "status": (
            "generation_complete_unjudged"
            if guard_passed
            else "generation_complete_cap_guard_failed_unjudged"
        ),
        "condition": "m0",
        "rows": len(existing),
        "truncations": truncations,
        "cap_guard_threshold": threshold,
        "cap_guard_passed": guard_passed,
        "model_revision": config["model_revision"],
        "generation_config_sha256": generation_config_sha256,
        "generation_implementation_sha256": implementation_sha256,
        "generations_sha256": sha256_file(final_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "evaluator_loaded": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


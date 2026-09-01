#!/usr/bin/env python3
"""Generate one fail-closed official safety condition without judging it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from gate_4b_coco_u1 import generate, read_manifest
from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.official_eval import resolve_final_safety_cap
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import ensure_latent_tokens
from train_4b_skip0_stage import load_model


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official_safety_endpoints.yaml")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-root", default=os.environ.get("WORK_ROOT"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.work_root:
        raise ValueError("--work-root or WORK_ROOT is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if args.condition not in config["conditions"]:
        raise ValueError(f"unregistered official condition: {args.condition}")
    condition = config["conditions"][args.condition]
    if condition.get("generation_status") != "authorized":
        raise ValueError(
            f"official condition is not authorized: {condition.get('generation_status')}"
        )
    evaluation_path = Path(config["evaluation_config"])
    evaluation = yaml.safe_load(evaluation_path.read_text())
    cap = int(
        condition.get(
            "max_new_tokens",
            resolve_final_safety_cap(evaluation, condition["branch"]),
        )
    )
    manifest_path = Path(config["manifest"])
    records = read_manifest(manifest_path)
    train_path = Path(condition["train_config"])
    train = yaml.safe_load(train_path.read_text())

    checkpoint_dir = Path(args.work_root) / "results" / condition["checkpoint_subdir"]
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    checkpoint = checkpoint_dir / "model_state.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if metadata["branch"] != condition["branch"]:
        raise ValueError("official checkpoint branch differs from condition")
    expected_stage = int(condition.get("checkpoint_stage", 3))
    if (
        metadata["completed_stage"] != expected_stage
        or metadata["k"] != condition["checkpoint_k"]
    ):
        raise ValueError("official checkpoint training stage/K differs from condition")
    if checkpoint_hash != condition["checkpoint_sha256"]:
        raise ValueError("official checkpoint hash differs from condition")
    if metadata["model_state_sha256"] != checkpoint_hash:
        raise ValueError("official checkpoint hash differs from metadata")
    if metadata["durability"]["status"] != "model_and_tokenizer_uploaded":
        raise ValueError("official checkpoint lacks durable model/tokenizer evidence")

    frozen = evaluation["sampling"]
    settings = {
        "serialization": evaluation["serialization"]["safety_format"][
            "explicit_cot" if condition["branch"] == "explicit_cot" else "coconut"
        ],
        "scaffold_kind": condition["scaffold_kind"],
        "max_new_tokens": cap,
        "do_sample": frozen["do_sample"],
        "temperature": frozen["temperature"],
        "top_p": frozen["top_p"],
        "top_k": frozen["top_k"],
        "stop_tokens": ["eos_token"],
    }
    generation_config_sha256 = sha256_json(settings)
    implementation_sha256 = sha256_json(
        {
            "generator": sha256_file(Path(__file__)),
            "shared_generation": sha256_file(Path("scripts/gate_4b_coco_u1.py")),
            "serialization": sha256_file(Path("src/mats_latent_safety/serialization.py")),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "generations.jsonl"
    partial_path = output_dir / "generations.partial.jsonl"
    if final_path.exists():
        raise FileExistsError("refusing to overwrite completed official generations")
    existing = []
    if partial_path.exists():
        existing = [json.loads(line) for line in partial_path.read_text().splitlines() if line]
    expected_prefix = [record["id"] for record in records[: len(existing)]]
    if [row.get("prompt_id") for row in existing] != expected_prefix:
        raise ValueError("official partial cache is not a manifest-order prefix")
    for row in existing:
        if (
            row.get("condition") != args.condition
            or row.get("checkpoint_sha256") != checkpoint_hash
            or row.get("generation_config_sha256") != generation_config_sha256
            or row.get("generation_implementation_sha256") != implementation_sha256
            or row.get("evaluator_payload") is not None
            or row.get("evaluator_score") is not None
        ):
            raise ValueError("official partial cache provenance differs from this run")

    tokenizer = AutoTokenizer.from_pretrained(train["model_id"], revision=train["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    model = load_model(train, tokenizer, markers, torch.device("cuda"))
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError("strict official checkpoint load returned key differences")

    code_revision = git_revision()
    for index, record in enumerate(records[len(existing) :], start=len(existing)):
        result = generate(
            model,
            tokenizer,
            markers,
            record["forbidden_prompt"],
            int(condition["k"]),
            settings,
            int(evaluation["seed"]) + index,
        )
        answer_tokens = len(
            tokenizer.encode(result["parsed_final_answer"], add_special_tokens=False)
        )
        thinking_tokens = len(
            tokenizer.encode(result["parsed_thinking"] or "", add_special_tokens=False)
        )
        prefix_tokens = len(
            tokenizer.encode(result["raw_serialized_prefix"], add_special_tokens=False)
        )
        input_tokens = prefix_tokens + (
            int(condition["k"]) + 2 if condition["scaffold_kind"] == "coconut_latent" else 0
        )
        row = {
            "schema_version": 1,
            "generation_purpose": config["generation_purpose"],
            "condition": args.condition,
            "prompt_id": record["id"],
            "prompt_sha256": record["sha256"],
            "category": record["category"],
            "model_id": train["model_id"],
            "model_revision": train["model_revision"],
            "code_revision": code_revision,
            "slurm_job_id": slurm_job_id(),
            "branch": condition["branch"],
            "checkpoint_sha256": checkpoint_hash,
            "k": int(condition["k"]),
            "generation_config_sha256": generation_config_sha256,
            "generation_implementation_sha256": implementation_sha256,
            "raw_serialized_input": result["raw_serialized_prefix"],
            "input_tokens": input_tokens,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "evaluator_payload": None,
            "evaluator_score": None,
            **result,
        }
        append_row(partial_path, row)
        existing.append(row)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "prompt_id": record["id"],
                    "generated_tokens": row["generated_tokens"],
                    "stop_reason": row["stop_reason"],
                }
            ),
            flush=True,
        )
    if len(existing) != len(records):
        raise RuntimeError("official generation cache is incomplete")
    partial_path.replace(final_path)
    summary = {
        "schema_version": 1,
        "status": "generation_complete_unjudged",
        "condition": args.condition,
        "rows": len(existing),
        "truncations": sum(bool(row["truncated"]) for row in existing),
        "checkpoint_sha256": checkpoint_hash,
        "generation_config_sha256": generation_config_sha256,
        "generation_implementation_sha256": implementation_sha256,
        "generations_sha256": sha256_file(final_path),
        "manifest_sha256": sha256_file(manifest_path),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "evaluator_loaded": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

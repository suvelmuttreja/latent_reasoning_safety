#!/usr/bin/env python3
"""Generate an exact-M0 native-chat control for the stage-1 coherence harness."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mats_latent_safety.hashing import sha256_file, sha256_json, sha256_text
from mats_latent_safety.parsing import parse_thinking_response
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import tokenize_native_chat_prompt


def summarize(rows: list[dict], max_new_tokens: int) -> dict:
    def one(group: list[dict]) -> dict:
        return {
            "outputs": len(group),
            "cap_hits": sum(
                row["stop_reason"] == "length"
                or row["generated_tokens"] >= max_new_tokens
                for row in group
            ),
            "missing_closing_think": sum(not row["has_closing_think"] for row in group),
            "cap_and_missing_closing_think": sum(
                (
                    row["stop_reason"] == "length"
                    or row["generated_tokens"] >= max_new_tokens
                )
                and not row["has_closing_think"]
                for row in group
            ),
            "eos_stops": sum(row["stop_reason"] == "eos_token" for row in group),
            "mean_generated_tokens": statistics.mean(
                row["generated_tokens"] for row in group
            ),
        }

    replicates = sorted({int(row["replicate"]) for row in rows})
    return {
        "overall": one(rows),
        "by_replicate": {
            str(replicate): one(
                [row for row in rows if int(row["replicate"]) == replicate]
            )
            for replicate in replicates
        },
    }


def validate_against_frozen_evaluation(config: dict, evaluation: dict) -> None:
    sampling = config["sampling"]
    frozen = evaluation["sampling"]
    for key in ("do_sample", "temperature", "top_p", "top_k"):
        if sampling[key] != frozen[key]:
            raise ValueError(f"control {key} differs from frozen evaluation config")
    if sampling["stop_tokens"] != evaluation["explicit_generation"]["stop_tokens"]:
        raise ValueError("control stop tokens differ from frozen explicit generation config")
    if sampling["max_new_tokens"] != evaluation["coconut_generation"]["answer_max_new_tokens"]:
        raise ValueError("control does not reproduce the shared 512-token gate cap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_4b_m0_coherence_control.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    evaluation_path = Path(config["evaluation_config"])
    evaluation = yaml.safe_load(evaluation_path.read_text())
    validate_against_frozen_evaluation(config, evaluation)
    manifest_path = Path(config["manifest"])
    manifest_payload = json.loads(manifest_path.read_text())
    records = manifest_payload["records"]
    if len(records) != 10:
        raise ValueError("registered coherence control requires exactly 10 prompts")
    for row in records:
        if sha256_text(row["prompt"]) != row["sha256"]:
            raise ValueError(f"coherence prompt hash mismatch: {row['id']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "m0_coherence_generations.jsonl"
    summary_path = output_dir / "m0_coherence_summary.json"
    existing: dict[str, dict] = {}
    if generations_path.exists():
        for line in generations_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[f"{row['replicate']}:{row['prompt_id']}"] = row

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    sampling = config["sampling"]
    max_new_tokens = int(sampling["max_new_tokens"])
    code_revision = git_revision()
    generation_config_sha256 = sha256_json(
        {
            "serialization": config["serialization"],
            "model_input": config["model_input"],
            "sampling": sampling,
            "seed_schedule": config["seed_schedule"],
        }
    )
    ordered_keys = [
        f"{replicate}:{row['id']}"
        for replicate in range(int(config["replicates"]))
        for row in records
    ]
    for replicate in range(int(config["replicates"])):
        for index, record in enumerate(records):
            key = f"{replicate}:{record['id']}"
            if key in existing:
                continue
            input_ids, serialized = tokenize_native_chat_prompt(
                tokenizer, record["prompt"]
            )
            batch = {
                "input_ids": torch.tensor([input_ids], device=model.device),
                "attention_mask": torch.ones(
                    (1, len(input_ids)), dtype=torch.long, device=model.device
                ),
            }
            sample_seed = (
                int(config["seed_schedule"]["base_seed"])
                + replicate * int(config["seed_schedule"]["replicate_stride"])
                + index
            )
            set_seed(sample_seed)
            started = time.perf_counter()
            with torch.inference_mode():
                sequence = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=bool(sampling["do_sample"]),
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            generated_ids = sequence[0, len(input_ids) :]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            parsed = parse_thinking_response(decoded)
            hit_eos = bool(
                len(generated_ids) and int(generated_ids[-1]) == tokenizer.eos_token_id
            )
            stop_reason = "eos_token" if hit_eos else "length"
            row = {
                "schema_version": 1,
                "source_condition": "exact_m0_native_chat_512",
                "replicate": replicate,
                "prompt_id": record["id"],
                "prompt_kind": record["kind"],
                "prompt_sha256": record["sha256"],
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "code_revision": code_revision,
                "slurm_job_id": slurm_job_id(),
                "sample_seed": sample_seed,
                "serialization": config["serialization"],
                "raw_serialized_prefix": serialized,
                "serialized_input_ids_sha256": sha256_json(input_ids),
                "raw_output": raw_output,
                "decoded_for_parsing": decoded,
                "parsed_thinking": parsed.thinking,
                "parsed_final_answer": parsed.final_answer,
                "closing_think_count": parsed.closing_delimiters,
                "has_closing_think": not parsed.missing_closing_delimiter,
                "input_tokens": len(input_ids),
                "generated_tokens": int(len(generated_ids)),
                "stop_reason": stop_reason,
                "truncated": stop_reason == "length",
                "seconds": seconds,
                "generation_config_sha256": generation_config_sha256,
            }
            existing[key] = row
            generations_path.write_text(
                "".join(
                    json.dumps(existing[item]) + "\n"
                    for item in ordered_keys
                    if item in existing
                )
            )
            print(
                json.dumps(
                    {
                        "replicate": replicate,
                        "prompt_id": record["id"],
                        "generated_tokens": len(generated_ids),
                        "stop_reason": stop_reason,
                        "has_closing_think": not parsed.missing_closing_delimiter,
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )

    rows = [existing[key] for key in ordered_keys]
    summary = {
        "schema_version": 1,
        "status": "technical_control_complete_pending_blind_human_scores",
        "next_stage_authorized": False,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
        "config_sha256": sha256_json(config),
        "evaluation_config_sha256": sha256_file(evaluation_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generation_config_sha256": generation_config_sha256,
        "generations_sha256": sha256_file(generations_path),
        "exact_m0_without_added_markers": True,
        "sampling_matches_frozen_evaluation": True,
        "stop_tokens_match_frozen_evaluation": True,
        "diagnostic_max_new_tokens": max_new_tokens,
        "registered_explicit_thinking_max_new_tokens": int(
            evaluation["explicit_generation"]["frozen_max_new_tokens"]
        ),
        "summary": summarize(rows, max_new_tokens),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

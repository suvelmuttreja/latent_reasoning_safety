#!/usr/bin/env python3
"""Generate the paired M0 dual-serialization StrongREJECT anchor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mats_latent_safety.hashing import sha256_json, sha256_text
from mats_latent_safety.parsing import is_incomplete_generation, parse_thinking_response
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    serialize_native_chat,
    tokenize_coconut_raw_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_anchor.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())["records"]
    sampling = config["sampling"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[f"{row['condition']}:{row['prompt_id']}"] = row

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    code_revision = git_revision()
    generation_config_sha256 = sha256_json(
        {"sampling": sampling, "paired_sampling": config["paired_sampling"]}
    )
    ordered_keys = [
        f"{condition}:{record['id']}" for record in manifest for condition in config["conditions"]
    ]

    for index, record in enumerate(manifest):
        prompt = record["forbidden_prompt"]
        paired_seed = int(config["paired_sampling"]["base_seed"]) + index
        for condition in config["conditions"]:
            key = f"{condition}:{record['id']}"
            prior = existing.get(key)
            if prior is not None:
                continue
            if condition == "native_chat":
                serialized = serialize_native_chat(tokenizer, prompt)
                batch = tokenizer(serialized, return_tensors="pt").to(model.device)
            elif condition == "coconut_raw_question":
                input_ids = tokenize_coconut_raw_prompt(tokenizer, prompt)
                serialized = tokenizer.decode(input_ids, skip_special_tokens=False)
                batch = {
                    "input_ids": torch.tensor([input_ids], device=model.device),
                    "attention_mask": torch.ones(
                        (1, len(input_ids)), dtype=torch.long, device=model.device
                    ),
                }
            else:
                raise ValueError(f"unsupported serialization condition: {condition}")
            set_seed(paired_seed)
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
            seconds = time.perf_counter() - started
            generated_ids = sequence[0, batch["input_ids"].shape[1] :]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed = parse_thinking_response(decoded)
            hit_eos = bool(len(generated_ids) and int(generated_ids[-1]) == tokenizer.eos_token_id)
            stop_reason = "eos_token" if hit_eos else "length"
            truncated = is_incomplete_generation(
                raw_output,
                stop_reason=stop_reason,
                generated_tokens=len(generated_ids),
                max_new_tokens=int(sampling["max_new_tokens"]),
            )
            row = {
                "schema_version": 1,
                "condition": condition,
                "prompt_id": record["id"],
                "category": record["category"],
                "prompt_sha256": sha256_text(prompt),
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "code_revision": code_revision,
                "slurm_job_id": slurm_job_id(),
                "paired_sampling_seed": paired_seed,
                "raw_serialized_input": serialized,
                "serialized_input_ids_sha256": sha256_json(batch["input_ids"][0].tolist()),
                "raw_output": raw_output,
                "decoded_for_parsing": decoded,
                "parsed_thinking": parsed.thinking,
                "parsed_final_answer": parsed.final_answer,
                "input_tokens": int(batch["input_ids"].shape[1]),
                "generated_tokens": int(len(generated_ids)),
                "answer_tokens": len(
                    tokenizer.encode(parsed.final_answer, add_special_tokens=False)
                ),
                "stop_reason": stop_reason,
                "truncated": truncated,
                "seconds": seconds,
                "generation_config_sha256": generation_config_sha256,
            }
            existing[key] = row
            output_path.write_text(
                "".join(json.dumps(existing[key]) + "\n" for key in ordered_keys if key in existing)
            )
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "prompt_id": record["id"],
                        "generated_tokens": len(generated_ids),
                        "truncated": truncated,
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

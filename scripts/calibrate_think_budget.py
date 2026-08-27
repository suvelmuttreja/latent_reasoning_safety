#!/usr/bin/env python3
"""Calibrate native Qwen thinking length on the frozen 20+12 prompt set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from mats_latent_safety.hashing import sha256_json, sha256_text
from mats_latent_safety.parsing import is_incomplete_generation, parse_thinking_response
from mats_latent_safety.runtime import git_revision, slurm_job_id


def load_prompts(manifest_dir: Path) -> list[dict]:
    gsm = json.loads((manifest_dir / "gsm8k_calibration_20.json").read_text())["records"]
    audit = json.loads((manifest_dir / "strongreject_audit_12.json").read_text())["records"]
    return [
        {"id": row["id"], "prompt": row["question"], "kind": "gsm8k"} for row in gsm
    ] + [
        {"id": row["id"], "prompt": row["forbidden_prompt"], "kind": "strongreject_audit"}
        for row in audit
    ]


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    explicit = config["explicit_generation"]
    sampling = config["sampling"]
    generation_parameters = {
        **sampling,
        "max_new_tokens": args.max_new_tokens,
        "stop_tokens": explicit["stop_tokens"],
    }
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        explicit["model_id"], revision=explicit["model_revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        explicit["model_id"],
        revision=explicit["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    prompts = load_prompts(Path(args.manifest_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            existing[row["prompt_id"]] = row

    generated = []
    code_revision = git_revision()
    run_start = time.perf_counter()
    total_generated_tokens = 0
    for prompt in prompts:
        prior = existing.get(prompt["id"])
        if prior is not None and not prior["truncated"]:
            generated.append(prior)
            continue
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tokenizer(serialized, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            sequence = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=sampling["do_sample"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        generated_ids = sequence[0, batch["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        decoded_for_parsing = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_thinking_response(decoded_for_parsing)
        hit_eos = bool(len(generated_ids) and int(generated_ids[-1]) == tokenizer.eos_token_id)
        stop_reason = "eos_token" if hit_eos else "length"
        truncated = is_incomplete_generation(
            raw_output,
            stop_reason=stop_reason,
            generated_tokens=len(generated_ids),
            max_new_tokens=args.max_new_tokens,
        )
        thinking_tokens = (
            len(tokenizer.encode(parsed.thinking, add_special_tokens=False))
            if parsed.thinking is not None
            else None
        )
        row = {
            "prompt_id": prompt["id"],
            "prompt_sha256": sha256_text(prompt["prompt"]),
            "kind": prompt["kind"],
            "model_id": explicit["model_id"],
            "model_revision": explicit["model_revision"],
            "code_revision": code_revision,
            "slurm_job_id": slurm_job_id(),
            "raw_serialized_input": serialized,
            "raw_output": raw_output,
            "decoded_for_parsing": decoded_for_parsing,
            "parsed_thinking": parsed.thinking,
            "parsed_final_answer": parsed.final_answer,
            "closing_delimiters": parsed.closing_delimiters,
            "input_tokens": int(batch["input_ids"].shape[1]),
            "thinking_tokens": thinking_tokens,
            "answer_tokens": len(tokenizer.encode(parsed.final_answer, add_special_tokens=False)),
            "generated_tokens": int(len(generated_ids)),
            "stop_reason": stop_reason,
            "truncated": truncated,
            "seconds": seconds,
            "tokens_per_second": len(generated_ids) / seconds,
            "max_new_tokens": args.max_new_tokens,
            "generation_config_sha256": sha256_json(generation_parameters),
        }
        existing[prompt["id"]] = row
        generated.append(row)
        total_generated_tokens += len(generated_ids)
        output_path.write_text("".join(json.dumps(existing[p["id"]]) + "\n" for p in prompts if p["id"] in existing))
        print(json.dumps({k: row[k] for k in ("prompt_id", "generated_tokens", "thinking_tokens", "truncated", "tokens_per_second")}), flush=True)

    lengths = [row["thinking_tokens"] for row in generated if row["thinking_tokens"] is not None]
    truncations = sum(bool(row["truncated"]) for row in generated)
    summary = {
        "model_id": explicit["model_id"],
        "model_revision": explicit["model_revision"],
        "prompts": len(generated),
        "max_new_tokens": args.max_new_tokens,
        "completed_thinking_lengths": len(lengths),
        "thinking_tokens_min": min(lengths) if lengths else None,
        "thinking_tokens_median": statistics.median(lengths) if lengths else None,
        "thinking_tokens_p95": percentile(lengths, 0.95) if lengths else None,
        "thinking_tokens_max": max(lengths) if lengths else None,
        "truncated": truncations,
        "truncation_rate": truncations / len(generated),
        "new_tokens_this_invocation": total_generated_tokens,
        "wall_seconds_this_invocation": time.perf_counter() - run_start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "code_revision": code_revision,
        "slurm_job_id": slurm_job_id(),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

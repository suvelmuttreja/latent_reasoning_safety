#!/usr/bin/env python3
"""Decode native token logits at each recurrent Coconut latent step.

This post-hoc readout measures decodability only. It neither intervenes on the
latent states nor establishes that decoded tokens faithfully describe the
computation used by the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from mats_latent_safety.hashing import sha256_file, sha256_json
from mats_latent_safety.runtime import git_revision, slurm_job_id
from mats_latent_safety.serialization import (
    build_coconut_question,
    ensure_latent_tokens,
    tokenize_coconut_raw_prompt,
    tokenize_native_chat_prompt,
)
from train_4b_skip0_stage import load_model


def read_records(path: Path, count: int) -> list[dict]:
    payload = json.loads(path.read_text())
    records = payload["records"]
    if len(records) < count:
        raise ValueError(f"{path} has fewer than {count} records")
    return records[:count]


def serialize(tokenizer, prompt: str, mode: str) -> tuple[list[int], str]:
    if mode == "coconut_raw_question":
        ids = tokenize_coconut_raw_prompt(tokenizer, prompt.strip())
        return ids, tokenizer.decode(ids, skip_special_tokens=False)
    if mode == "native_qwen_chat_with_latent_scaffold":
        return tokenize_native_chat_prompt(tokenizer, prompt.strip())
    raise ValueError(f"unsupported serialization: {mode}")


def clean_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


@torch.no_grad()
def read_prompt(model, tokenizer, markers, input_ids: list[int], k: int, top_k: int) -> list[dict]:
    scaffold = build_coconut_question(input_ids, markers, k)
    device = next(model.parameters()).device
    calls: list[dict] = []
    output_head = model.base_causallm.get_output_embeddings()

    def capture(_module, _args, output):
        hidden = output.hidden_states[-1][0, -1, :]
        native_logits = output.logits[0, -1, :]
        projected_logits = output_head(hidden)
        max_error = float((native_logits.float() - projected_logits.float()).abs().max().item())
        values, token_ids = torch.topk(native_logits.float(), k=top_k)
        log_normalizer = torch.logsumexp(native_logits.float(), dim=-1)
        calls.append(
            {
                "hidden_l2": float(torch.linalg.vector_norm(hidden.float()).item()),
                "native_projection_max_abs_error": max_error,
                "top_tokens": [
                    {
                        "rank": rank + 1,
                        "token_id": int(token_id),
                        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                        "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                        "logit": float(value),
                        "probability": float(torch.exp(value - log_normalizer).item()),
                    }
                    for rank, (value, token_id) in enumerate(
                        zip(values.tolist(), token_ids.tolist(), strict=True)
                    )
                ],
            }
        )

    handle = model.base_causallm.register_forward_hook(capture)
    try:
        model(torch.tensor([scaffold["input_ids"]], device=device))
    finally:
        handle.remove()
    if len(calls) != k + 1:
        raise RuntimeError(f"expected {k + 1} base-model calls, observed {len(calls)}")
    latent_calls = calls[:k]
    for depth, row in enumerate(latent_calls, start=1):
        row["latent_depth"] = depth
    return latent_calls


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Post-hoc token-mode latent readout",
        "",
        "Native next-token projections at each recurrent latent step. These are",
        "decodability observations only, not faithfulness or monitorability claims.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['task']} — {row['prompt_id']}",
                "",
                "| Depth | Top decoded tokens (rank order) | Top-1 probability | Hidden L2 |",
                "|---:|---|---:|---:|",
            ]
        )
        for readout in row["readouts"]:
            tokens = ", ".join(
                f"`{clean_markdown(token['decoded'])}`" for token in readout["top_tokens"]
            )
            lines.append(
                f"| {readout['latent_depth']} | {tokens} | "
                f"{readout['top_tokens'][0]['probability']:.4f} | "
                f"{readout['hidden_l2']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc_token_mode_readout.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    train_path = Path(config["train_config"])
    train = yaml.safe_load(train_path.read_text())
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / "model_state.pt"
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    checkpoint_hash = sha256_file(checkpoint_path)
    expected = config["checkpoint"]
    if metadata["branch"] != "coconut_skip0":
        raise ValueError("readout requires a Coconut checkpoint")
    if metadata["completed_stage"] != expected["stage"] or metadata["k"] != expected["k"]:
        raise ValueError("checkpoint stage/K differs from readout config")
    if checkpoint_hash != expected["model_state_sha256"]:
        raise ValueError("checkpoint hash differs from frozen readout config")
    if metadata["model_state_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint hash differs from stage metadata")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty readout output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(train["model_id"], revision=train["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    markers = ensure_latent_tokens(tokenizer)
    model = load_model(train, tokenizer, markers, torch.device("cuda"))
    state = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("strict endpoint load returned key differences")
    model.eval()

    rows = []
    for task in config["tasks"]:
        manifest_path = Path(task["manifest"])
        records = read_records(manifest_path, int(task["prompt_count"]))
        for record in records:
            prompt = record[task["prompt_field"]]
            prefix_ids, rendered = serialize(tokenizer, prompt, task["serialization"])
            rows.append(
                {
                    "task": task["name"],
                    "prompt_id": record["id"],
                    "prompt_sha256": record.get("question_sha256", record["sha256"]),
                    "serialization": task["serialization"],
                    "serialized_prefix_sha256": sha256_json(prefix_ids),
                    "rendered_prefix": rendered,
                    "readouts": read_prompt(
                        model,
                        tokenizer,
                        markers,
                        prefix_ids,
                        int(expected["k"]),
                        int(config["top_k"]),
                    ),
                }
            )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_role": "posthoc_exploratory_decodability_only",
        "faithfulness_claimed": False,
        "monitorability_claimed": False,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": sha256_json(config),
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "prompt_selection": "first_n_in_each_preexisting_frozen_manifest",
        "rows": rows,
    }
    json_path = output_dir / "token_mode_readout.json"
    markdown_path = output_dir / "token_mode_readout.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(markdown_path, rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "prompts": len(rows),
        "latent_depths_per_prompt": int(expected["k"]),
        "top_k": int(config["top_k"]),
        "maximum_native_projection_error": max(
            readout["native_projection_max_abs_error"]
            for row in rows
            for readout in row["readouts"]
        ),
        "json_sha256": sha256_file(json_path),
        "markdown_sha256": sha256_file(markdown_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

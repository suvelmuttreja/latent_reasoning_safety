#!/usr/bin/env python3
"""Audit a gated public wrapper state dict before any strict=False load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mats_latent_safety.coconut import StandardCoconut
from mats_latent_safety.serialization import ensure_latent_tokens
from mats_latent_safety.state_audit import audit_state_dict


def state_mapping(value):
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint root must be a dict, got {type(value).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        nested = value.get(key)
        if isinstance(nested, dict) and nested:
            return nested, key
    return value, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument(
        "--base-revision", default="768f209d9ea81521153ed38c47d515654e938aea"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = hf_hub_download(args.repo_id, args.filename, revision=args.revision)
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    actual, nested_key = state_mapping(raw)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    marker_ids = ensure_latent_tokens(tokenizer)
    config = AutoConfig.from_pretrained(args.base_model, revision=args.base_revision)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(config)
        base.resize_token_embeddings(len(tokenizer))
        expected_model = StandardCoconut(
            base,
            latent_token_id=marker_ids["latent"],
            start_latent_id=marker_ids["start"],
            end_latent_id=marker_ids["end"],
            eos_token_id=tokenizer.eos_token_id,
        )
    expected = expected_model.state_dict()
    audit = audit_state_dict(expected, actual)
    result = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "filename": args.filename,
        "download_path": str(path),
        "checkpoint_nested_key": nested_key,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "expected_key_count": len(expected),
        "actual_key_count": len(actual),
        "audit": audit.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not audit.clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


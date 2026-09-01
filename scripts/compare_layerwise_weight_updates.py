#!/usr/bin/env python3
"""Describe layer-wise endpoint weight updates relative to the pinned M0.

This is a post-hoc descriptive analysis. Parameter movement does not localize
the cause of a behavioral change and is not a mechanistic attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from mats_latent_safety.hashing import sha256_file
from mats_latent_safety.runtime import git_revision


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def group_name(base_key: str) -> str:
    match = LAYER_RE.match(base_key)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    if base_key.startswith("model.embed_tokens."):
        return "embeddings"
    if base_key.startswith("model.norm."):
        return "final_norm"
    if base_key.startswith("lm_head."):
        return "lm_head"
    return "other"


def component_name(base_key: str) -> str:
    match = LAYER_RE.match(base_key)
    suffix = match.group(2) if match else base_key
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ):
        if name in suffix:
            return name
    if base_key.startswith("model.embed_tokens."):
        return "embeddings"
    if base_key.startswith("model.norm."):
        return "final_norm"
    if base_key.startswith("lm_head."):
        return "lm_head"
    return "other"


def comparable_views(
    base: torch.Tensor, cot: torch.Tensor, coconut: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if cot.shape != coconut.shape:
        raise ValueError(f"endpoint shape mismatch: {cot.shape} != {coconut.shape}")
    if base.shape == cot.shape:
        return base, cot, coconut, "exact_shape"
    if (
        base.ndim >= 1
        and cot.ndim == base.ndim
        and cot.shape[0] != base.shape[0]
        and cot.shape[1:] == base.shape[1:]
    ):
        # Qwen's public matrix contains padded vocabulary rows, while training
        # resize_token_embeddings() uses tokenizer length plus the three marker
        # rows. Neither tail has a one-to-one counterpart. Compare only the
        # shared token-ID prefix and expose both original shapes in the output.
        common_rows = min(base.shape[0], cot.shape[0])
        return (
            base[:common_rows],
            cot[:common_rows],
            coconut[:common_rows],
            "common_token_id_prefix",
        )
    raise ValueError(f"M0/endpoint shape mismatch: {base.shape} != {cot.shape}")


def tensor_moments(
    base: torch.Tensor,
    cot: torch.Tensor,
    coconut: torch.Tensor,
    *,
    chunk_elements: int,
) -> dict[str, float | int]:
    base_flat = base.reshape(-1)
    cot_flat = cot.reshape(-1)
    coconut_flat = coconut.reshape(-1)
    sums = defaultdict(float)
    for start in range(0, base_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, base_flat.numel())
        b = base_flat[start:stop].float()
        c = cot_flat[start:stop].float()
        k = coconut_flat[start:stop].float()
        dc = c - b
        dk = k - b
        sums["base_sq"] += float(torch.dot(b, b).item())
        sums["cot_sq"] += float(torch.dot(dc, dc).item())
        sums["coconut_sq"] += float(torch.dot(dk, dk).item())
        sums["cross"] += float(torch.dot(dc, dk).item())
    return {"elements": base_flat.numel(), **sums}


def summarize(name: str, moments: dict[str, float | int]) -> dict[str, float | int | str | None]:
    elements = int(moments["elements"])
    base_sq = float(moments["base_sq"])
    cot_sq = float(moments["cot_sq"])
    coconut_sq = float(moments["coconut_sq"])
    cross = float(moments["cross"])
    base_l2 = math.sqrt(base_sq)
    cot_l2 = math.sqrt(cot_sq)
    coconut_l2 = math.sqrt(coconut_sq)
    denominator = cot_l2 * coconut_l2
    return {
        "group": name,
        "elements": elements,
        "m0_l2": base_l2,
        "cot_update_l2": cot_l2,
        "coconut_update_l2": coconut_l2,
        "cot_relative_l2": cot_l2 / base_l2 if base_l2 else None,
        "coconut_relative_l2": coconut_l2 / base_l2 if base_l2 else None,
        "coconut_to_cot_update_ratio": coconut_l2 / cot_l2 if cot_l2 else None,
        "update_cosine": cross / denominator if denominator else None,
        "cot_update_rms": cot_l2 / math.sqrt(elements),
        "coconut_update_rms": coconut_l2 / math.sqrt(elements),
    }


def add_moments(target: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key in ("elements", "base_sq", "cot_sq", "coconut_sq", "cross"):
        target[key] = target.get(key, 0) + source[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-snapshot", required=True)
    parser.add_argument("--cot-checkpoint", required=True)
    parser.add_argument("--coconut-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    args = parser.parse_args()

    m0_dir = Path(args.m0_snapshot)
    cot_path = Path(args.cot_checkpoint)
    coconut_path = Path(args.coconut_checkpoint)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty weight-diff output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((m0_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    cot_state = torch.load(cot_path, map_location="cpu", weights_only=True, mmap=True)
    coconut_state = torch.load(coconut_path, map_location="cpu", weights_only=True, mmap=True)
    endpoint_keys = sorted(key for key in cot_state if key.startswith("base_causallm."))
    if set(endpoint_keys) != {
        key for key in coconut_state if key.startswith("base_causallm.")
    }:
        raise ValueError("CoT and Coconut endpoint parameter keys differ")

    per_parameter = []
    grouped: dict[str, dict[str, float | int]] = defaultdict(dict)
    component_grouped: dict[str, dict[str, float | int]] = defaultdict(dict)
    excluded = []
    with ExitStack() as stack:
        shard_handles = {
            shard: stack.enter_context(safe_open(m0_dir / shard, framework="pt", device="cpu"))
            for shard in sorted(set(weight_map.values()))
        }
        for checkpoint_key in endpoint_keys:
            base_key = checkpoint_key.removeprefix("base_causallm.")
            if base_key not in weight_map:
                excluded.append(
                    {"checkpoint_key": checkpoint_key, "reason": "no_public_m0_parameter"}
                )
                continue
            base = shard_handles[weight_map[base_key]].get_tensor(base_key)
            cot = cot_state[checkpoint_key]
            coconut = coconut_state[checkpoint_key]
            base, cot, coconut, comparison = comparable_views(base, cot, coconut)
            moments = tensor_moments(
                base, cot, coconut, chunk_elements=args.chunk_elements
            )
            group = group_name(base_key)
            component = component_name(base_key)
            add_moments(grouped[group], moments)
            add_moments(component_grouped[component], moments)
            row = summarize(base_key, moments)
            row.update(
                {
                    "checkpoint_key": checkpoint_key,
                    "comparison": comparison,
                    "endpoint_shape": list(cot_state[checkpoint_key].shape),
                    "compared_shape": list(base.shape),
                }
            )
            per_parameter.append(row)

    layer_rows = [summarize(name, grouped[name]) for name in sorted(grouped)]
    component_rows = [
        summarize(name, component_grouped[name]) for name in sorted(component_grouped)
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_role": "posthoc_descriptive_not_mechanistic_attribution",
        "interpretation_guard": (
            "parameter movement does not identify where behavioral damage lives"
        ),
        "m0_snapshot": str(m0_dir),
        "cot_checkpoint": str(cot_path),
        "coconut_checkpoint": str(coconut_path),
        "cot_checkpoint_sha256": sha256_file(cot_path),
        "coconut_checkpoint_sha256": sha256_file(coconut_path),
        "code_revision": git_revision(),
        "mismatched_vocabulary_matrices_compare_common_token_id_prefix_only": True,
        "excluded_parameters": excluded,
        "overall": summarize("overall", {
            key: sum(float(row.get(key, 0)) for row in grouped.values())
            for key in ("elements", "base_sq", "cot_sq", "coconut_sq", "cross")
        }),
        "layers": layer_rows,
        "components": component_rows,
        "parameters": per_parameter,
    }
    json_path = output_dir / "layerwise_weight_updates.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    for filename, rows in (
        ("layerwise_weight_updates.csv", layer_rows),
        ("component_weight_updates.csv", component_rows),
    ):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

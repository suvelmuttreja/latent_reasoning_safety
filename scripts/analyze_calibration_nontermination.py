#!/usr/bin/env python3
"""Compute content-blind structural diagnostics for safety calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zlib
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from mats_latent_safety.hashing import sha256_file
from mats_latent_safety.nontermination import exact_suffix_cycle
from mats_latent_safety.nontermination import unique_ngram_ratio


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    generations_path = Path(args.generations)
    if sha256_file(generations_path) != config["source_generations_sha256"]:
        raise ValueError("generation cache differs from preregistered source hash")
    rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"]
    )
    metrics = config["metrics"]
    suffix = metrics["exact_suffix_cycle"]
    tail_tokens = int(metrics["token_4gram_unique_ratio_last_tokens"])
    records = []
    grouped = {}
    for row in rows:
        raw = row["raw_output"]
        token_ids = tokenizer.encode(raw, add_special_tokens=False)
        tail = token_ids[-tail_tokens:]
        cycle = exact_suffix_cycle(
            token_ids,
            int(suffix["max_period_tokens"]),
            int(suffix["minimum_repeats"]),
            int(suffix["minimum_covered_tokens"]),
        )
        encoded = raw.encode()
        numeric = {
            "k": int(row["k"]),
            "stop_reason": row["stop_reason"],
            "generated_tokens": int(row["generated_tokens"]),
            "closing_think_delimiters": raw.count("</think>"),
            "token_4gram_unique_ratio_last_2048": unique_ngram_ratio(tail),
            "utf8_zlib_compression_ratio": len(zlib.compress(encoded)) / max(1, len(encoded)),
            "exact_suffix_cycle": cycle,
            "structural_nontermination_flag": row["stop_reason"] == "length" and cycle is not None,
        }
        key = (numeric["k"], numeric["stop_reason"])
        grouped.setdefault(key, []).append(numeric)
        if row["stop_reason"] == "length":
            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "raw_output_sha256": hashlib.sha256(encoded).hexdigest(),
                    **numeric,
                }
            )
    by_group = {}
    for (k, stop_reason), group in sorted(grouped.items()):
        by_group[f"k{k}_{stop_reason}"] = {
            "rows": len(group),
            "closing_think_present": sum(item["closing_think_delimiters"] > 0 for item in group),
            "structural_nontermination_flags": sum(
                item["structural_nontermination_flag"] for item in group
            ),
            "token_4gram_unique_ratio_last_2048": summarize(
                [item["token_4gram_unique_ratio_last_2048"] for item in group]
            ),
            "utf8_zlib_compression_ratio": summarize(
                [item["utf8_zlib_compression_ratio"] for item in group]
            ),
        }
    result = {
        "schema_version": 1,
        "status": "content_blind_structural_diagnostic_complete",
        "source_generations_sha256": sha256_file(generations_path),
        "config_sha256": sha256_file(config_path),
        "evaluator_loaded": False,
        "raw_content_emitted": False,
        "by_group": by_group,
        "length_stop_records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "length_stop_records"}, indent=2))


if __name__ == "__main__":
    main()

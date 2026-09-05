#!/usr/bin/env python3
"""Measure the configured judge's input visibility without loading model weights."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-snapshot", required=True, type=Path)
    parser.add_argument(
        "--generations-root",
        required=True,
        type=Path,
        help="Directory containing m0/cot_u3/coco_u3_k6 generations.jsonl",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_snapshot),
        local_files_only=True,
        padding_side="left",
        truncation_side="left",
    )
    conditions, hashes = {}, {}
    for condition in ("m0", "cot_u3", "coco_u3_k6"):
        path = args.generations_root / condition / "generations.jsonl"
        raw = path.read_bytes()
        hashes[condition] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        counts = [
            {
                "prompt_id": row["prompt_id"],
                "judge_tokens": len(
                    tokenizer(row["parsed_final_answer"], truncation=False)["input_ids"]
                ),
            }
            for row in rows
        ]
        if len({row["prompt_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate prompt IDs")
        lengths = [row["judge_tokens"] for row in counts]
        conditions[condition] = {
            "rows": len(rows),
            "over_512": sum(n > 512 for n in lengths),
            "maximum": max(lengths),
            "mean": statistics.mean(lengths),
            "rows_by_id": counts,
        }
    print(
        json.dumps(
            {
                "tokenizer_snapshot": str(args.tokenizer_snapshot),
                "conditions": conditions,
                "source_sha256": hashes,
                "tokenization": "upstream tokenizer defaults, without truncation",
                "audit_script_sha256": (
                    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    if Path(__file__).is_file()
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

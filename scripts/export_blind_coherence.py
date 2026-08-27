#!/usr/bin/env python3
"""Export unlabeled coherence records with a reproducible shuffle key."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="JSONL generation records")
    parser.add_argument("output", help="Blinded JSONL output")
    parser.add_argument("--key-output", required=True, help="Private label-key JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    random.Random(args.seed).shuffle(rows)
    blinded, key = [], []
    for index, row in enumerate(rows):
        blind_id = f"blind-{index:04d}"
        output = row["parsed_final_answer"]
        blinded.append({"blind_id": blind_id, "output": output, "score_0_to_2": None})
        key.append(
            {
                "blind_id": blind_id,
                "prompt_id": row["prompt_id"],
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "k": row.get("k"),
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            }
        )
    Path(args.output).write_text("".join(json.dumps(row) + "\n" for row in blinded))
    Path(args.key_output).write_text(json.dumps(key, indent=2) + "\n")


if __name__ == "__main__":
    main()


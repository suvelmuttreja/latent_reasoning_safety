#!/usr/bin/env python3
"""Combine stage and M0 coherence outputs into one condition-blind packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from mats_latent_safety.hashing import sha256_file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-input", required=True)
    parser.add_argument("--m0-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-output", required=True)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    stage_path = Path(args.stage_input)
    m0_path = Path(args.m0_input)
    stage = read_jsonl(stage_path)
    m0 = read_jsonl(m0_path)
    if len(stage) != 20 or len(m0) != 20:
        raise ValueError("combined blind packet requires 20 stage and 20 M0 rows")
    tagged = [(f"stage1_k{row['k']}", row) for row in stage] + [
        (f"m0_replicate_{row['replicate']}", row) for row in m0
    ]
    random.Random(args.seed).shuffle(tagged)
    blinded, key = [], []
    for index, (source_condition, row) in enumerate(tagged):
        blind_id = f"harness-blind-{index:04d}"
        output = row["parsed_final_answer"]
        blinded.append({"blind_id": blind_id, "output": output, "score_0_to_2": None})
        key.append(
            {
                "blind_id": blind_id,
                "source_condition": source_condition,
                "prompt_id": row["prompt_id"],
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            }
        )
    output_path = Path(args.output)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in blinded))
    key_payload = {
        "schema_version": 1,
        "shuffle_seed": args.seed,
        "stage_input_sha256": sha256_file(stage_path),
        "m0_input_sha256": sha256_file(m0_path),
        "records": key,
    }
    Path(args.key_output).write_text(json.dumps(key_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

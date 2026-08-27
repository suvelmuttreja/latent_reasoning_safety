#!/usr/bin/env python3
"""Render the first 20 canonical clean-GSM8K records for required inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mats_latent_safety.data import clean_gsm8k_row
from mats_latent_safety.hashing import sha256_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm-parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = pd.read_parquet(args.gsm_parquet).iloc[:20]
    records = []
    for source_index, row in rows.iterrows():
        example = clean_gsm8k_row(str(row.question), str(row.answer))
        records.append(
            {
                "source_index": int(source_index),
                "question": example["question"],
                "steps": example["steps"],
                "answer": example["answer"],
                "latent_stage_format": {
                    "stage_1": "question + <|start-latent|> + 2 latent positions + <|end-latent|> + steps[1:] + answer",
                    "stage_2": "question + <|start-latent|> + 4 latent positions + <|end-latent|> + steps[2:] + answer",
                    "stage_3": "question + <|start-latent|> + 6 latent positions + <|end-latent|> + steps[3:] + answer",
                },
                "sha256": sha256_json(example),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "inspection_required",
        "note": "Schema is access-independent smoke preprocessing; final 4B schema remains subject to public-wrapper audit.",
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} processed records to {output}")


if __name__ == "__main__":
    main()

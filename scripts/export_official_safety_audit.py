#!/usr/bin/env python3
"""Export the frozen endpoint audit subset without condition labels or judge scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from mats_latent_safety.hashing import sha256_file
from mats_latent_safety.official_eval import validate_official_safety_rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official_safety_human_audit.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    official_path = Path(config["official_config"])
    official = yaml.safe_load(official_path.read_text())
    full_manifest_path = Path(official["manifest"])
    full_manifest = json.loads(full_manifest_path.read_text())["records"]
    audit_manifest_path = Path(config["audit_manifest"])
    audit_manifest = json.loads(audit_manifest_path.read_text())["records"]
    audit_by_id = {row["id"]: row for row in audit_manifest}
    if len(audit_by_id) != 12:
        raise ValueError("official human audit requires exactly 12 unique prompts")

    source_rows = []
    generation_hashes = {}
    for condition in config["conditions"]:
        generation_path = Path(config["generation_files"][condition])
        rows = validate_official_safety_rows(
            read_jsonl(generation_path), full_manifest, condition
        )
        generation_hashes[condition] = sha256_file(generation_path)
        by_id = {row["prompt_id"]: row for row in rows}
        for prompt_id, audit_row in audit_by_id.items():
            generation = by_id[prompt_id]
            if generation["prompt_sha256"] != audit_row["sha256"]:
                raise ValueError("audit prompt hash differs from official generation")
            source_rows.append(
                {
                    "source_condition": condition,
                    "prompt_id": prompt_id,
                    "prompt_sha256": audit_row["sha256"],
                    "category": audit_row["category"],
                    "forbidden_prompt": audit_row["forbidden_prompt"],
                    "full_response": generation["raw_output"],
                    "parsed_final_answer": generation["parsed_final_answer"],
                }
            )
    if len(source_rows) != 24:
        raise ValueError("official human audit requires 12 rows from each condition")
    random.Random(int(config["shuffle_seed"])).shuffle(source_rows)

    packet = []
    key_records = []
    for index, row in enumerate(source_rows):
        blind_id = f"endpoint-audit-blind-{index:04d}"
        displayed = {
            "blind_id": blind_id,
            "category": row["category"],
            "forbidden_prompt": row["forbidden_prompt"],
            "full_response": row["full_response"],
            "parsed_final_answer": row["parsed_final_answer"],
            "human_label": None,
            "coherent": None,
            "note": "",
        }
        packet.append(displayed)
        key_records.append(
            {
                "blind_id": blind_id,
                "source_condition": row["source_condition"],
                "prompt_id": row["prompt_id"],
                "prompt_sha256": row["prompt_sha256"],
                "displayed_sha256": hashlib.sha256(
                    json.dumps(displayed, sort_keys=True).encode()
                ).hexdigest(),
            }
        )

    packet_path = Path(config["outputs"]["packet"])
    key_path = Path(config["outputs"]["key"])
    if packet_path.exists() or key_path.exists():
        raise FileExistsError("refusing to overwrite an official human-audit packet")
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(packet_path, packet)
    key = {
        "schema_version": 1,
        "status": "sealed_pending_condition_blind_human_labels",
        "shuffle_seed": int(config["shuffle_seed"]),
        "packet_sha256": sha256_file(packet_path),
        "config_sha256": sha256_file(config_path),
        "official_config_sha256": sha256_file(official_path),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "generation_sha256_by_condition": generation_hashes,
        "records": key_records,
    }
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    print(json.dumps({"packet": str(packet_path), "rows": 24, "packet_sha256": key["packet_sha256"], "key": str(key_path)}))


if __name__ == "__main__":
    main()


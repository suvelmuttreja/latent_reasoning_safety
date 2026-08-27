#!/usr/bin/env python3
"""Non-secret HF identity, revision, and one-byte read-access audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from mats_latent_safety.runtime import git_revision, slurm_job_id


COSTCO_REPOS = [
    "Costco666/qwen3-4b-gsm8k-coconut-full-k6",
    "Costco666/qwen3-4b-gsm8k-coconut-skip0-k6",
    "Costco666/qwen3-4b-gsm8k-coconut-full-k12",
    "Costco666/qwen3-4b-gsm8k-coconut-skip0-k12",
    "Costco666/qwen3-4b-gsm8k-coconut-full-k12-intermediates",
    "Costco666/qwen3-4b-gsm8k-coconut-skip0-k6-intermediates",
    "Costco666/qwen3-4b-gsm8k-coconut-skip0-k12-intermediates",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    api = HfApi()
    identity = api.whoami()["name"]
    rows = []
    for repo_id in COSTCO_REPOS:
        info = api.model_info(repo_id)
        filenames = [item.rfilename for item in info.siblings]
        probe_file = next((name for name in filenames if name != "README.md"), filenames[0])
        response = requests.get(
            hf_hub_url(repo_id, probe_file, revision=info.sha),
            headers={**build_hf_headers(token=True), "Range": "bytes=0-0"},
            stream=True,
            timeout=30,
        )
        rows.append(
            {
                "repo_id": repo_id,
                "revision": info.sha,
                "gated": info.gated,
                "probe_file": probe_file,
                "probe_status": response.status_code,
                "readable": response.status_code in (200, 206),
                "siblings": filenames,
            }
        )
        response.close()
    result = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "authenticated_identity": identity,
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "all_costco_readable": all(row["readable"] for row in rows),
        "repositories": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

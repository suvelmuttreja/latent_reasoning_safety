#!/usr/bin/env python3
"""Persist resolved HF cache revisions and byte counts without reading tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import constants, scan_cache_dir

from mats_latent_safety.runtime import git_revision, slurm_job_id


REQUIRED = {
    "model": {
        "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-4B-Thinking-2507": "768f209d9ea81521153ed38c47d515654e938aea",
        "qylu4156/strongreject-15k-v1": "4bd893d32390d2cace4f067dc2e3ef5294fd78a2",
        "google/gemma-2b": "9cf48e52b224239de00d483ec8eb84fb8d0f3a3a",
    },
    "dataset": {
        "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = scan_cache_dir()
    rows = []
    present = set()
    for repo in cache.repos:
        expected = REQUIRED.get(repo.repo_type, {}).get(repo.repo_id)
        if expected is None:
            continue
        revisions = []
        for revision in repo.revisions:
            revisions.append(
                {
                    "commit_hash": revision.commit_hash,
                    "snapshot_path": str(revision.snapshot_path),
                    "size_on_disk": revision.size_on_disk,
                    "files": len(revision.files),
                    "matches_pin": revision.commit_hash == expected,
                }
            )
            if revision.commit_hash == expected:
                present.add((repo.repo_type, repo.repo_id))
        rows.append(
            {
                "repo_type": repo.repo_type,
                "repo_id": repo.repo_id,
                "expected_revision": expected,
                "size_on_disk": repo.size_on_disk,
                "revisions": revisions,
            }
        )
    required_pairs = {
        (repo_type, repo_id) for repo_type, repos in REQUIRED.items() for repo_id in repos
    }
    result = {
        "schema_version": 1,
        "code_revision": git_revision(),
        "slurm_job_id": slurm_job_id(),
        "cache_dir": constants.HF_HUB_CACHE,
        "repos": rows,
        "present_at_pinned_revision": sorted(f"{kind}:{repo}" for kind, repo in present),
        "not_yet_cached": sorted(f"{kind}:{repo}" for kind, repo in required_pairs - present),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

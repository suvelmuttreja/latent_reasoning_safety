#!/usr/bin/env python3
"""Verify private HF checkpoint-repository upload and clean download."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    api = HfApi()
    identity = api.whoami()["name"]
    info = api.model_info(args.repo_id)
    if not info.private:
        raise RuntimeError(f"durable checkpoint repository must be private: {args.repo_id}")
    now = datetime.now(timezone.utc).isoformat()
    remote_path = f"durability-smoke/{now[:10]}-{uuid.uuid4().hex}.json"
    artifact = {
        "schema_version": 1,
        "purpose": "MATS latent-safety durable checkpoint repository write/read verification",
        "created_at": now,
        "repo_id": args.repo_id,
        "contains_secrets": False,
    }
    with tempfile.TemporaryDirectory(prefix="mats-hf-smoke-") as temporary:
        source = Path(temporary) / "artifact.json"
        source.write_text(json.dumps(artifact, sort_keys=True) + "\n")
        source_hash = digest(source)
        commit = api.upload_file(
            path_or_fileobj=source,
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Verify MATS checkpoint durability path",
        )
        downloaded = Path(
            hf_hub_download(
                args.repo_id,
                remote_path,
                repo_type="model",
                revision=commit.oid,
                force_download=True,
                local_dir=Path(temporary) / "download",
            )
        )
        downloaded_hash = digest(downloaded)
    if source_hash != downloaded_hash:
        raise RuntimeError("uploaded and downloaded durability artifact hashes differ")
    result = {
        "status": "passed",
        "authenticated_identity": identity,
        "repo_id": args.repo_id,
        "repo_private": info.private,
        "remote_path": remote_path,
        "commit_revision": commit.oid,
        "source_sha256": source_hash,
        "downloaded_sha256": downloaded_hash,
        "hashes_match": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


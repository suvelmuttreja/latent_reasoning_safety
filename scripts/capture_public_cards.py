#!/usr/bin/env python3
"""Capture pinned Costco model cards after manual approval lands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", default="configs/pins.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    pins = json.loads(Path(args.pins).read_text())["huggingface"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    captured = []
    for repo_id in (
        "Costco666/qwen3-4b-gsm8k-coconut-full-k6",
        "Costco666/qwen3-4b-gsm8k-coconut-skip0-k6",
    ):
        revision = pins[repo_id]
        source = Path(hf_hub_download(repo_id, "README.md", revision=revision))
        target = output / f"{repo_id.rsplit('/', 1)[-1]}-{revision}.md"
        target.write_bytes(source.read_bytes())
        captured.append({"repo_id": repo_id, "revision": revision, "path": str(target)})
    (output / "index.json").write_text(json.dumps(captured, indent=2) + "\n")
    print(json.dumps(captured, indent=2))


if __name__ == "__main__":
    main()

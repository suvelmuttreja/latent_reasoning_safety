#!/usr/bin/env python3
"""Verify the committed evidence and recompute reported analyses without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOTS = ("artifacts/discovery", "configs", "manifests", "project_plan")


def evidence_files(root: Path) -> set[str]:
    """Include all scientific inputs/outputs, excluding explanatory README files."""
    return {
        str(p.relative_to(root))
        for directory in FROZEN_ROOTS
        for p in (root / directory).rglob("*")
        if p.is_file()
        and p.name != "README.md"
        and not {"__pycache__", "tokenizer"}.intersection(p.parts)
        and p.name != ".DS_Store"
    }


def verify_checksums(root: Path) -> int:
    expected = {}
    for line in (root / "artifacts/checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in expected or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid checksum entry: {name}")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe checksum path: {name}")
        expected[name] = digest
    actual = evidence_files(root)
    if actual != set(expected):
        raise ValueError(
            f"Evidence inventory differs: added={sorted(actual - set(expected))}, "
            f"missing={sorted(set(expected) - actual)}"
        )
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    return len(expected)


def verify_json(root: Path) -> int:
    count = 0
    for directory in (*FROZEN_ROOTS, "writeup"):
        for p in (root / directory).rglob("*"):
            if {"__pycache__", "tokenizer"}.intersection(p.parts):
                continue
            if p.suffix == ".json":
                json.loads(p.read_text())
                count += 1
            elif p.suffix == ".jsonl":
                for number, line in enumerate(p.read_text().splitlines(), 1):
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON: {p}:{number}") from exc
                count += 1
    return count


def verify_links(root: Path) -> int:
    # Raw model outputs and historical notebooks contain model-authored URLs;
    # validate maintained documentation, not the text of experimental stimuli.
    pages = [root / "README.md", root / "THIRD_PARTY_NOTICES.md"]
    pages += sorted((root / "docs").glob("*.md"))
    pages += [p for p in root.glob("*/README.md") if p.parent.name != "vendor"]
    count = 0
    for page in pages:
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            if re.match(r"[a-z]+://|mailto:|#", target):
                continue
            path = unquote(target.split("#", 1)[0].strip("<>"))
            if not (page.parent / path).exists():
                raise ValueError(f"Broken local link in {page.relative_to(root)}: {target}")
            count += 1
    return count


def verify_shell(root: Path) -> int:
    paths = sorted((root / "scripts").glob("*.sh"))
    paths += sorted((root / "scripts/slurm").glob("*.sbatch"))
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    return len(paths)


def compare_analysis(actual: object, expected: object, path: str = "audit") -> None:
    """Allow only floating-point roundoff across Python versions/platforms."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Analysis keys differ at {path}")
        for key in expected:
            compare_analysis(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Analysis lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_analysis(left, right, f"{path}[{index}]")
    elif isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Analysis number differs at {path}: {actual} != {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Analysis value differs at {path}")


def verify_recomputation(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="mats-verify-") as directory:
        work = Path(directory)
        spec = importlib.util.spec_from_file_location(
            "writeup_build", root / "writeup/build_full_writeup.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUTPUT = work / "FULL_WRITEUP.built.md"
        module.main()
        if module.OUTPUT.read_bytes() != (root / "writeup/FULL_WRITEUP.built.md").read_bytes():
            raise ValueError("Rebuilt write-up differs from the committed Markdown")
        output = work / "recomputed_checks.json"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/audit_existing_analyses.py"),
                "--output",
                str(output),
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        expected = json.loads((root / "writeup/analysis_audit/recomputed_checks.json").read_text())
        compare_analysis(json.loads(output.read_text()), expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-recompute", action="store_true", help="only validate saved files")
    args = parser.parse_args()
    print(f"Verified {verify_checksums(ROOT)} evidence checksums")
    print(f"Parsed {verify_json(ROOT)} JSON/JSONL files")
    print(f"Checked {verify_links(ROOT)} local documentation links")
    print(f"Checked {verify_shell(ROOT)} shell/SLURM scripts")
    if not args.skip_recompute:
        verify_recomputation(ROOT)
        print("Write-up and numerical analysis reproduce the committed outputs")


if __name__ == "__main__":
    main()

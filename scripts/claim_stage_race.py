#!/usr/bin/env python3
"""Atomically select one queued hardware path and cancel its rivals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_hardware_race.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    current = int(os.environ["SLURM_JOB_ID"])
    jobs = {name: int(job_id) for name, job_id in config["jobs"].items() if job_id is not None}
    if current not in jobs.values():
        raise ValueError(f"current job {current} is absent from the frozen race registry")

    claim = Path(config["claim_directory"])
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError as error:
        winner_path = claim / "winner.json"
        winner = winner_path.read_text().strip() if winner_path.exists() else "unknown"
        raise RuntimeError(f"hardware race already claimed; winner={winner}") from error

    cancellations = []
    for name, job_id in jobs.items():
        if job_id == current:
            continue
        result = subprocess.run(
            ["scancel", str(job_id)],
            check=False,
            capture_output=True,
            text=True,
        )
        cancellations.append(
            {
                "name": name,
                "job_id": job_id,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to cancel rival job {job_id}: {result.stderr.strip()}")

    record = {
        "schema_version": 1,
        "race_id": config["race_id"],
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "winner_job_id": current,
        "winner_name": next(name for name, job_id in jobs.items() if job_id == current),
        "cancelled_rivals": cancellations,
        "rule": config["rule"],
    }
    serialized = json.dumps(record, indent=2) + "\n"
    (claim / "winner.json").write_text(serialized)
    record_path = Path(config["record_path"])
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(serialized)
    print(serialized, flush=True)


if __name__ == "__main__":
    main()

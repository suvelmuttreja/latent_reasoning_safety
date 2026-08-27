"""Runtime provenance helpers."""

from __future__ import annotations

import os
import subprocess


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def slurm_job_id() -> str | None:
    return os.getenv("SLURM_JOB_ID")


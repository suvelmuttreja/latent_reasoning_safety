"""Runtime provenance for Git checkouts and committed HPC source exports."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_revision() -> str:
    """Resolve this project's revision, independently of the process's cwd.

    ``sync.sh`` exports committed code without Git metadata and supplies the
    revision marker. It takes precedence over a stale .git directory left by
    earlier syncs. Ordinary checkouts report local modifications explicitly.
    """
    marker = ROOT / ".source-revision"
    if marker.exists():
        revision = marker.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Invalid source revision in {marker}")
        return revision
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    return f"{revision}-dirty" if dirty else revision


def slurm_job_id() -> str | None:
    return os.getenv("SLURM_JOB_ID")

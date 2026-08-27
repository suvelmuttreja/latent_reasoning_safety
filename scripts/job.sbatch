#!/usr/bin/env bash
# Deprecated compatibility entrypoint. Use a task-specific script in
# scripts/slurm so GPU type, time, and scientific command are reviewable.
set -euo pipefail
echo "Use scripts/slurm/smoke_0_6b.sbatch, strongreject_smoke.sbatch, or calibrate_m0.sbatch" >&2
exit 2

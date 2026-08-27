#!/usr/bin/env bash
# Push local code to Discovery scratch. Code only — no data, checkpoints, or caches.
# Usage: ./scripts/sync.sh [--pull]
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"
REMOTE_HOST="discovery"
REMOTE="/scratch1/muttreja/mats_latent_safety/code/"

EXCLUDES=(
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc'
  --exclude '.venv' --exclude '.ipynb_checkpoints'
  --exclude 'logs/' --exclude 'data/' --exclude 'checkpoints/'
  --exclude '.DS_Store' --exclude 'wandb/'
)

if [[ "${1:-}" == "--pull" ]]; then
  # Bring results back. Deliberately no --delete: never clobber local work.
  rsync -avz --progress \
    "$REMOTE_HOST:/scratch1/muttreja/mats_latent_safety/logs/" "${LOCAL}logs/"
else
  rsync -avz --delete "${EXCLUDES[@]}" "$LOCAL" "$REMOTE_HOST:$REMOTE"
fi

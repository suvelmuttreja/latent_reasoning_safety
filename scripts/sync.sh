#!/usr/bin/env bash
# Sync canonical local code to home1, or durable result copies back from scratch1.
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"
REMOTE_HOST="discovery"
REMOTE_CODE="/home1/muttreja/mats_latent_safety/"
REMOTE_WORK="/scratch1/muttreja/mats_latent_safety/"

case "${1:-}" in
  --push)
    ssh "$REMOTE_HOST" 'mkdir -p /home1/$USER/mats_latent_safety /scratch1/$USER/mats_latent_safety/{cache,data,logs,results,checkpoints,tmp}'
    rsync -avz --delete-delay \
      --exclude '.venv/' --exclude '.uv-cache/' --exclude 'vendor/' \
      --exclude 'artifacts/' --exclude '.DS_Store' --exclude '.pytest_cache/' \
      --exclude '.ruff_cache/' --exclude '__pycache__/' \
      --exclude '*.pyc' --exclude 'data/' --exclude 'checkpoints/' \
      "$LOCAL" "$REMOTE_HOST:$REMOTE_CODE"
    ;;
  --pull)
    mkdir -p "${LOCAL}artifacts/discovery/logs" "${LOCAL}artifacts/discovery/results"
    rsync -avz "$REMOTE_HOST:${REMOTE_WORK}logs/" "${LOCAL}artifacts/discovery/logs/"
    rsync -avz "$REMOTE_HOST:${REMOTE_WORK}results/" "${LOCAL}artifacts/discovery/results/"
    ;;
  *)
    echo "usage: $0 --push | --pull" >&2
    exit 2
    ;;
esac

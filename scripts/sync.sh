#!/usr/bin/env bash
# Export committed code to Discovery, or pull compact evidence to the local checkout.
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-discovery}"
mode="${1:-}"
if [[ "$mode" != --push && "$mode" != --pull ]] || [[ $# -gt 2 ]] || \
   [[ $# -eq 2 && "$2" != --dry-run ]]; then
  echo "usage: $0 --push | --pull [--dry-run]" >&2
  exit 2
fi
rsync_options=(-avz)
dry_run=false
if [[ "${2:-}" == --dry-run ]]; then
  rsync_options+=(--dry-run)
  dry_run=true
fi

# Query the remote account; local macOS and HPC usernames need not match.
REMOTE_USER="${REMOTE_USER:-$(ssh "$REMOTE_HOST" 'id -un')}"
REMOTE_CODE="${REMOTE_CODE:-/home1/$REMOTE_USER/mats_latent_safety}"
REMOTE_WORK="${REMOTE_WORK:-/scratch1/$REMOTE_USER/mats_latent_safety}"
REMOTE_CODE="${REMOTE_CODE%/}"
REMOTE_WORK="${REMOTE_WORK%/}"
# Paths enter a remote shell command. Require simple absolute project paths.
for path in "$REMOTE_CODE" "$REMOTE_WORK"; do
  if [[ ! "$path" =~ ^/[-a-zA-Z0-9_.]+/[-a-zA-Z0-9_.]+/[-a-zA-Z0-9_./]+$ ]] || \
     [[ "$path" == *'/../'* || "$path" == */.. || "$path" == *'/./'* || \
        "$path" == */. || "$path" == /project2/* ]]; then
    echo "unsupported remote project path: $path" >&2
    exit 2
  fi
done

case "$mode" in
  --push)
    if [[ -n "$(git -C "$LOCAL" status --porcelain)" ]]; then
      echo "commit the working tree before syncing; remote provenance must identify exact code" >&2
      exit 2
    fi
    snapshot="$(mktemp -d)"
    trap 'rm -rf "$snapshot"' EXIT
    # An archive excludes untracked credentials, caches, and .git/config.
    git -C "$LOCAL" archive HEAD | tar -x -C "$snapshot"
    git -C "$LOCAL" rev-parse HEAD > "$snapshot/.source-revision"
    if [[ "$dry_run" == false ]]; then
      ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_CODE' '$REMOTE_WORK'/cache '$REMOTE_WORK'/data '$REMOTE_WORK'/logs '$REMOTE_WORK'/results '$REMOTE_WORK'/checkpoints '$REMOTE_WORK'/tmp"
    fi
    rsync "${rsync_options[@]}" --delete-delay \
      --exclude '.git/' --exclude '.venv/' --exclude '.uv-cache/' --exclude 'vendor/' \
      --exclude 'artifacts/' --exclude '.DS_Store' --exclude '.pytest_cache/' \
      --exclude '.ruff_cache/' --exclude '__pycache__/' --exclude '*.pyc' \
      --exclude 'data/' --exclude 'checkpoints/' --exclude 'results/' --exclude 'logs/' \
      --exclude '.env' --exclude '.env.*' --exclude '*.token' --exclude 'token' \
      --exclude '*.pem' --exclude '*.key' \
      "$snapshot/" "$REMOTE_HOST:$REMOTE_CODE/"
    ;;
  --pull)
    if [[ "$dry_run" == false ]]; then
      mkdir -p "$LOCAL/artifacts/discovery/logs" "$LOCAL/artifacts/discovery/results"
    fi
    rsync "${rsync_options[@]}" "$REMOTE_HOST:$REMOTE_WORK/logs/" "$LOCAL/artifacts/discovery/logs/"
    rsync "${rsync_options[@]}" \
      --exclude '*.pt' --exclude '*.bin' --exclude '*.safetensors' \
      --exclude 'checkpoint*' --exclude 'tokenizer/' --exclude '*.partial.jsonl' \
      "$REMOTE_HOST:$REMOTE_WORK/results/" "$LOCAL/artifacts/discovery/results/"
    ;;
esac

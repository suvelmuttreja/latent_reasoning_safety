#!/usr/bin/env bash
# Reproducible Discovery environment setup. Run on a login node after --push.
set -euo pipefail

CODE_ROOT="/home1/$USER/mats_latent_safety"
WORK_ROOT="/scratch1/$USER/mats_latent_safety"
MIN_HOME_FREE_GIB=20

if [[ ! -f "$CODE_ROOT/uv.lock" ]]; then
  echo "missing pinned lockfile: $CODE_ROOT/uv.lock" >&2
  exit 2
fi

quota_line="$(myquota | awk -v root="/home1/$USER" '$1 == root {print $2, $4}')"
read -r home_used home_limit <<<"$quota_line"
if [[ -z "${home_used:-}" || -z "${home_limit:-}" ]]; then
  echo "could not parse home1 usage from myquota; stop for user direction" >&2
  exit 3
fi
home_free="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$home_used" "$home_limit")"
if ! python3 -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$home_free" "$MIN_HOME_FREE_GIB"; then
  echo "home1 has only ${home_free} GiB free; at least ${MIN_HOME_FREE_GIB} GiB is required" >&2
  echo "environment was not created or relocated; ask the user how to proceed" >&2
  exit 4
fi

mkdir -p "$WORK_ROOT"/{cache/uv,cache/huggingface,data,logs,results,checkpoints,tmp}
module purge
module load gcc/13.3.0 cuda/12.6.3

export UV_CACHE_DIR="$WORK_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$CODE_ROOT/.venv"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export HF_TOKEN_PATH="/home1/$USER/.cache/huggingface/token"

cd "$CODE_ROOT"
uv sync --frozen

echo "environment=$CODE_ROOT/.venv"
echo "home1_free_before_sync_gib=$home_free"
"$CODE_ROOT/.venv/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)'
myquota

#!/usr/bin/env bash
# Shared environment for Discovery jobs. Source; do not execute directly.
set -euo pipefail

CODE_ROOT="/home1/$USER/mats_latent_safety"
WORK_ROOT="/scratch1/$USER/mats_latent_safety"

module purge
module load gcc/13.3.0 cuda/12.6.3
export UV_CACHE_DIR="$WORK_ROOT/cache/uv"
export HF_HOME="$WORK_ROOT/cache/huggingface"
export HF_TOKEN_PATH="/home1/$USER/.cache/huggingface/token"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS=1
source "$CODE_ROOT/.venv/bin/activate"
cd "$CODE_ROOT"

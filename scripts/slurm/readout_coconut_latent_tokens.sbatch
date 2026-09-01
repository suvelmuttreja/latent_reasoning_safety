#!/usr/bin/env bash
#SBATCH --job-name=mats-latent-readout
#SBATCH --account=piray_1101
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
OUTPUT="$WORK_ROOT/results/posthoc_token_mode_readout"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/posthoc_token_mode_readout"
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
trap copy_results EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/readout_coconut_latent_tokens.py \
  --config configs/posthoc_token_mode_readout.yaml \
  --checkpoint-dir "$WORK_ROOT/results/fallback_4b_skip0/stage3" \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT"/* "$HOME_COPY"/*

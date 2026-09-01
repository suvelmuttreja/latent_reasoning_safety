#!/usr/bin/env bash
#SBATCH --job-name=mats-layer-weight-diff
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
M0_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea"
OUTPUT="$WORK_ROOT/results/posthoc_layerwise_weight_diff"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/posthoc_layerwise_weight_diff"
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
trap copy_results EXIT
srun python -u scripts/compare_layerwise_weight_updates.py \
  --m0-snapshot "$M0_SNAPSHOT" \
  --cot-checkpoint "$WORK_ROOT/results/matched_4b_cot/stage3/model_state.pt" \
  --coconut-checkpoint "$WORK_ROOT/results/fallback_4b_skip0/stage3/model_state.pt" \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT"/* "$HOME_COPY"/*

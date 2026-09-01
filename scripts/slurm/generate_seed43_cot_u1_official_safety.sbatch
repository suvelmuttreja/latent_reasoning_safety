#!/usr/bin/env bash
#SBATCH --job-name=mats-seed43-official-u1
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
OUTPUT="$WORK_ROOT/results/matched_4b_cot_seed43/official_safety_u1"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/matched_4b_cot_seed43/official_safety_u1"
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
trap copy_results EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/generate_official_safety.py \
  --config configs/seed43_cot_u1_official_safety.yaml \
  --condition cot_seed43_u1 \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT/generations.jsonl" "$HOME_COPY/generations.jsonl"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

#!/usr/bin/env bash
#SBATCH --job-name=mats-official-coco-u3-k6
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
OUTPUT="$WORK_ROOT/results/official_safety/coco_u3_k6"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/official_safety/coco_u3_k6"
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
if [[ ! -d "$OUTPUT" && -d "$HOME_COPY" ]]; then
  mkdir -p "$OUTPUT"
  rsync -a "$HOME_COPY/" "$OUTPUT/"
fi
trap copy_results EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/generate_official_safety.py \
  --config configs/official_safety_endpoints.yaml \
  --condition coco_u3_k6 \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT/generations.jsonl" "$HOME_COPY/generations.jsonl"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

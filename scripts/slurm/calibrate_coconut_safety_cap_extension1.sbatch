#!/usr/bin/env bash
#SBATCH --job-name=mats-coco-cap-ext1
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
PARENT="$WORK_ROOT/results/fallback_4b_skip0/safety_cap_calibration_stage1/generations.jsonl"
OUTPUT="$WORK_ROOT/results/fallback_4b_skip0/safety_cap_calibration_stage1_extension1"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/fallback_4b_skip0/safety_cap_calibration_stage1_extension1"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/calibrate_coconut_safety_cap.py \
  --config configs/coconut_safety_cap_calibration_extension1.yaml \
  --checkpoint-dir "$WORK_ROOT/results/fallback_4b_skip0/stage1" \
  --prior-generations "$PARENT" \
  --output-dir "$OUTPUT"
mkdir -p "$HOME_COPY"
rsync -a "$OUTPUT/" "$HOME_COPY/"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

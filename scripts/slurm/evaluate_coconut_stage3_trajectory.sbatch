#!/usr/bin/env bash
#SBATCH --job-name=mats-coco-s3-trajectory
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
OUTPUT="$WORK_ROOT/results/fallback_4b_skip0/trajectory_stage3"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/fallback_4b_skip0/trajectory_stage3"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/evaluate_coconut_stage_trajectory.py \
  --stage 3 \
  --checkpoint-dir "$WORK_ROOT/results/fallback_4b_skip0/stage3" \
  --output-dir "$OUTPUT"
mkdir -p "$HOME_COPY"
rsync -a "$OUTPUT/" "$HOME_COPY/"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

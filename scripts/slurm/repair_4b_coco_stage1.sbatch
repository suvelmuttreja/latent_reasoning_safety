#!/usr/bin/env bash
#SBATCH --job-name=mats-repair-coco-s1
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
srun python -u scripts/repair_4b_stage_finalization.py \
  --config configs/fallback_4b_skip0.yaml \
  --stage 1 \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --checkpoint-dir "$WORK_ROOT/results/fallback_4b_skip0/stage1" \
  --training-log "$WORK_ROOT/logs/mats-fallback-4b-coco-s1-a40-11413196.out" \
  --training-job-id 11413196 \
  --training-code-revision bbf86b152a7d4a565a655e67811a509530ebb32c

#!/usr/bin/env bash
#SBATCH --job-name=mats-s1-coco-validate
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/validate_0_6b.py \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --output-dir "$WORK_ROOT/results/s1/validation_0_6b"


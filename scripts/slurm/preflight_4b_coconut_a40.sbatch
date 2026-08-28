#!/usr/bin/env bash
#SBATCH --job-name=mats-s1-4b-coco-a40
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/preflight_4b_coconut.py \
  --config configs/preflight_4b_coconut_a40.yaml \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --output "$WORK_ROOT/results/s1/preflight_4b_coconut_a40.json"


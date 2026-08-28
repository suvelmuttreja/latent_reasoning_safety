#!/usr/bin/env bash
#SBATCH --job-name=mats-score-format-anchor
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/score_m0_format_anchor.py \
  --generations "$WORK_ROOT/results/m0_format_anchor/generations.jsonl" \
  --output "$WORK_ROOT/results/m0_format_anchor/scores.json"

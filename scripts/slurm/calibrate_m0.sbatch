#!/usr/bin/env bash
#SBATCH --job-name=mats-s1-m0-cal
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
# Inference-only 4B calibration fits an A40; the later full-training memory
# preflight remains separately constrained to A100-80GB.
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Resumable: the Python driver persists each prompt before moving to the next.
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/calibrate_think_budget.py \
  --max-new-tokens 16000 \
  --output "$WORK_ROOT/results/s1/m0_think_calibration.jsonl"

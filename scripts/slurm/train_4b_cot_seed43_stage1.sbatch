#!/usr/bin/env bash
#SBATCH --job-name=mats-cot-seed43-u1
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
if [[ "${INLINE_GATE_ACK:-}" != "coco-u1-inline-gate-passed" ]]; then
  echo "the completed Coconut u1 gate acknowledgement is required" >&2
  exit 2
fi
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/train_4b_skip0_stage.py \
  --config configs/matched_4b_cot_seed43_u1.yaml \
  --stage 1 \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --output-root "$WORK_ROOT/results/matched_4b_cot_seed43" \
  --acknowledge-inline-gate "$INLINE_GATE_ACK"

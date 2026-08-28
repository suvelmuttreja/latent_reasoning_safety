#!/usr/bin/env bash
#SBATCH --job-name=mats-matched-4b-cot-s1-a40
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
if [[ "${EARLY_COT_ACK:-}" != "time-pressure-early-cot-stage1-authorized" ]]; then
  echo "time-pressure early-CoT exception is not acknowledged; refusing submission" >&2
  exit 2
fi
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/train_4b_skip0_stage.py \
  --config configs/matched_4b_cot.yaml \
  --stage 1 \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --output-root "$WORK_ROOT/results/matched_4b_cot" \
  --acknowledge-inline-gate "$EARLY_COT_ACK"

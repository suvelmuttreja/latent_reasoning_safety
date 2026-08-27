#!/usr/bin/env bash
#SBATCH --job-name=mats-matched-4b-cot-s3
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
if [[ "${INLINE_GATE_ACK:-}" != "coco-u1-inline-gate-passed" ]]; then
  echo "coco_u1 in-line gate is not acknowledged; refusing submission" >&2
  exit 2
fi
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/train_4b_skip0_stage.py \
  --config configs/matched_4b_cot.yaml \
  --stage 3 \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --output-root "$WORK_ROOT/results/matched_4b_cot" \
  --resume-dir "$WORK_ROOT/results/matched_4b_cot/stage2" \
  --acknowledge-inline-gate "$INLINE_GATE_ACK"


#!/usr/bin/env bash
#SBATCH --job-name=mats-gate-4b-coco-u1-v2
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
OUTPUT="$WORK_ROOT/results/fallback_4b_skip0/gate_stage1_v2"
srun python -u scripts/gate_4b_coco_u1.py \
  --checkpoint-dir "$WORK_ROOT/results/fallback_4b_skip0/stage1" \
  --output-dir "$OUTPUT"
python scripts/export_blind_coherence.py \
  "$OUTPUT/coherence_generations.jsonl" \
  "$OUTPUT/coherence_blinded.jsonl" \
  --key-output "$OUTPUT/coherence_blind_key.json" \
  --seed 42

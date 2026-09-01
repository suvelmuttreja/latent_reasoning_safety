#!/usr/bin/env bash
#SBATCH --job-name=mats-score-seed43-u1
#SBATCH --account=piray_1101
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
cd "$CODE_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/score_cached_strongreject.py \
  --generations artifacts/discovery/results/matched_4b_cot_seed43/official_safety_u1/generations.jsonl \
  --manifest manifests/strongreject_small_60.json \
  --output artifacts/discovery/results/matched_4b_cot_seed43/scores/cot_seed43_u1.json \
  --evaluation-config configs/evaluation.yaml \
  --official-condition cot_seed43_u1 \
  --authorization-config configs/seed43_cot_u1_safety_scoring.yaml
sha256sum artifacts/discovery/results/matched_4b_cot_seed43/scores/cot_seed43_u1.json

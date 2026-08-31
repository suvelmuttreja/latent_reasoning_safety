#!/usr/bin/env bash
#SBATCH --job-name=mats-score-official-coco-u3-k6
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
  --generations artifacts/discovery/results/official_safety/coco_u3_k6/generations.jsonl \
  --manifest manifests/strongreject_small_60.json \
  --output artifacts/discovery/results/official_safety/scores/coco_u3_k6.json \
  --evaluation-config configs/evaluation.yaml \
  --official-condition coco_u3_k6 \
  --authorization-config configs/official_safety_scoring.yaml
sha256sum artifacts/discovery/results/official_safety/scores/coco_u3_k6.json


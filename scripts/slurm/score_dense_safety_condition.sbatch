#!/usr/bin/env bash
#SBATCH --job-name=mats-score-dense-safety
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
: "${CONDITION:?submit with an authorized dense trajectory condition}"
case "$CONDITION" in
  cot_u1|cot_u2)
    ;;
  *)
    echo "unsupported authorized dense scoring condition: $CONDITION" >&2
    exit 2
    ;;
esac
cd "$CODE_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/score_cached_strongreject.py \
  --generations "artifacts/discovery/results/dense_safety/$CONDITION/generations.jsonl" \
  --manifest manifests/strongreject_small_60.json \
  --output "artifacts/discovery/results/dense_safety/scores/$CONDITION.json" \
  --evaluation-config configs/evaluation.yaml \
  --official-condition "$CONDITION" \
  --authorization-config configs/dense_safety_scoring.yaml
sha256sum "artifacts/discovery/results/dense_safety/scores/$CONDITION.json"

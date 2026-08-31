#!/usr/bin/env bash
#SBATCH --job-name=mats-compare-official-safety
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
cd "$CODE_ROOT"
python scripts/compare_official_safety_scores.py \
  --config configs/official_safety_scoring.yaml
sha256sum artifacts/discovery/results/official_safety/scores/paired_comparison.json

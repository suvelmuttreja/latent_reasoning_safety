#!/usr/bin/env bash
#SBATCH --job-name=mats-coco-nonterm
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
SOURCE="$WORK_ROOT/results/fallback_4b_skip0/safety_cap_calibration_stage3/generations.jsonl"
OUTPUT="$CODE_ROOT/artifacts/discovery/results/fallback_4b_skip0/safety_nontermination_stage3.json"
srun python -u scripts/analyze_calibration_nontermination.py \
  --config configs/coconut_safety_nontermination_diagnostic.yaml \
  --generations "$SOURCE" \
  --output "$OUTPUT"
sha256sum "$OUTPUT"

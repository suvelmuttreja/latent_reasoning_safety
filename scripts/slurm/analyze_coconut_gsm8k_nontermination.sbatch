#!/usr/bin/env bash
#SBATCH --job-name=mats-coco-gsm-nonterm
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
SOURCE="$WORK_ROOT/results/fallback_4b_skip0/trajectory_stage3_gsm8k_cap1024/gsm8k.jsonl"
OUTPUT="$CODE_ROOT/artifacts/discovery/results/fallback_4b_skip0/gsm8k_nontermination_stage3.json"
srun python -u scripts/analyze_calibration_nontermination.py \
  --config configs/coconut_gsm8k_nontermination_diagnostic.yaml \
  --generations "$SOURCE" \
  --output "$OUTPUT"
sha256sum "$OUTPUT"

#!/usr/bin/env bash
#SBATCH --job-name=mats-repair-cot-s1
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
srun python -u scripts/repair_4b_stage_finalization.py \
  --config configs/matched_4b_cot.yaml \
  --stage 1 \
  --gsm-train "$WORK_ROOT/data/sources/gsm8k-train.parquet" \
  --checkpoint-dir "$WORK_ROOT/results/matched_4b_cot/stage1" \
  --training-log "$WORK_ROOT/logs/mats-matched-4b-cot-s1-a40-11413535.out" \
  --training-job-id 11413535 \
  --training-code-revision 0f938f38b2b31a1e103979d72a98ae87135ae940

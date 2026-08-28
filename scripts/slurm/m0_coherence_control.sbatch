#!/usr/bin/env bash
#SBATCH --job-name=mats-m0-coh-control
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
OUTPUT="$WORK_ROOT/results/fallback_4b_skip0/gate_stage1_v2_m0_coherence_control"
mkdir -p "$OUTPUT"
srun python -u scripts/run_m0_coherence_control.py \
  --output-dir "$OUTPUT"
python scripts/export_coherence_harness_blind.py \
  --stage-input "$WORK_ROOT/results/fallback_4b_skip0/gate_stage1_v2/coherence_generations.jsonl" \
  --m0-input "$OUTPUT/m0_coherence_generations.jsonl" \
  --output "$OUTPUT/coherence_harness_blinded.jsonl" \
  --key-output "$OUTPUT/coherence_harness_blind_key.json" \
  --seed 4242

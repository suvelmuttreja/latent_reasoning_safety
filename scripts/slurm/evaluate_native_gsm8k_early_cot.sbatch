#!/usr/bin/env bash
#SBATCH --job-name=mats-early-cot-gsm
#SBATCH --account=piray_1101
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
: "${CONDITION:?submit with --export=ALL,CONDITION=cot_u1 or cot_u2}"
case "$CONDITION" in
  cot_u1|cot_u2) ;;
  *) echo "unsupported condition: $CONDITION" >&2; exit 2 ;;
esac
OUTPUT="$WORK_ROOT/results/native_gsm8k_controls/$CONDITION"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/native_gsm8k_controls/$CONDITION"
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
if [[ ! -d "$OUTPUT" && -d "$HOME_COPY" ]]; then
  mkdir -p "$OUTPUT"
  rsync -a "$HOME_COPY/" "$OUTPUT/"
fi
trap copy_results EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/evaluate_native_gsm8k_control.py \
  --config configs/native_gsm8k_early_cot_controls.yaml \
  --condition "$CONDITION" \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT/generations.jsonl" "$HOME_COPY/generations.jsonl"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

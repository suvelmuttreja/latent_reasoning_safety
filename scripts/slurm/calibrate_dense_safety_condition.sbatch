#!/usr/bin/env bash
#SBATCH --job-name=mats-dense-safety-cap
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
: "${CONDITION:?submit with CONDITION=cot_u1, cot_u2, or coco_u2}"
case "$CONDITION" in
  cot_u1)
    CONFIG="configs/dense_cot_u1_safety_cap.yaml"
    CHECKPOINT="$WORK_ROOT/results/matched_4b_cot/stage1"
    OUTPUT="$WORK_ROOT/results/dense_safety/cot_u1_cap"
    HOME_COPY="$CODE_ROOT/artifacts/discovery/results/dense_safety/cot_u1_cap"
    ;;
  cot_u2)
    CONFIG="configs/dense_cot_u2_safety_cap.yaml"
    CHECKPOINT="$WORK_ROOT/results/matched_4b_cot/stage2"
    OUTPUT="$WORK_ROOT/results/dense_safety/cot_u2_cap"
    HOME_COPY="$CODE_ROOT/artifacts/discovery/results/dense_safety/cot_u2_cap"
    ;;
  coco_u2)
    CONFIG="configs/dense_coco_u2_safety_cap.yaml"
    CHECKPOINT="$WORK_ROOT/results/fallback_4b_skip0/stage2"
    OUTPUT="$WORK_ROOT/results/dense_safety/coco_u2_cap"
    HOME_COPY="$CODE_ROOT/artifacts/discovery/results/dense_safety/coco_u2_cap"
    ;;
  *)
    echo "unsupported dense safety condition: $CONDITION" >&2
    exit 2
    ;;
esac
copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}
trap copy_results EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/calibrate_coconut_safety_cap.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CHECKPOINT" \
  --output-dir "$OUTPUT"
copy_results
trap - EXIT
sha256sum "$OUTPUT/generations.jsonl" "$HOME_COPY/generations.jsonl"
sha256sum "$OUTPUT/summary.json" "$HOME_COPY/summary.json"

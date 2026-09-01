#!/usr/bin/env bash
#SBATCH --job-name=mats-official-coco-a40-rescue
#SBATCH --account=piray_1101
#SBATCH --partition=debug
#SBATCH --constraint=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
cd "$CODE_ROOT"

ATTEMPT="${1:-1}"
MAX_ATTEMPTS=3
A100_PRIMARY=11523096
A100_FALLBACK=11528116
A100_PRIMARY_AUDIT=11533445
A100_FALLBACK_AUDIT=11533446
OUTPUT="$WORK_ROOT/results/official_safety/coco_u3_k6"
HOME_COPY="$CODE_ROOT/artifacts/discovery/results/official_safety/coco_u3_k6"
CLAIM_DIR="$WORK_ROOT/results/official_safety/.coco_u3_k6_a40_rescue_claim"
RACE_LOSER=0

if [[ -f "$OUTPUT/generations.jsonl" && -f "$OUTPUT/summary.json" ]]; then
  echo "official Coconut cache already complete; rescue is a no-op"
  exit 0
fi

mkdir -p "$(dirname "$CLAIM_DIR")"
if ! mkdir "$CLAIM_DIR" 2>/dev/null; then
  echo "another A40 rescue candidate claimed the official Coconut path"
  exit 0
fi
printf '%s\n' "$SLURM_JOB_ID" > "$CLAIM_DIR/job_id"
while read -r peer_job; do
  if [[ -n "$peer_job" && "$peer_job" != "$SLURM_JOB_ID" ]]; then
    scancel "$peer_job" || true
  fi
done < <(squeue -h -n mats-official-coco-a40-rescue -o '%A')

if [[ "$ATTEMPT" == "1" ]]; then
  FALLBACK_STATE="$(squeue -h -j "$A100_FALLBACK" -o '%T' | head -n 1)"
  if [[ "$FALLBACK_STATE" == "RUNNING" || "$FALLBACK_STATE" == "COMPLETING" ]]; then
    echo "A100 fallback started first; A40 rescue exits without opening output"
    RACE_LOSER=1
    rm -f "$CLAIM_DIR/job_id"
    rmdir "$CLAIM_DIR"
    exit 0
  fi
  if [[ "$FALLBACK_STATE" == "PENDING" ]]; then
    scontrol hold "$A100_FALLBACK"
    FALLBACK_STATE="$(squeue -h -j "$A100_FALLBACK" -o '%T' | head -n 1)"
    if [[ "$FALLBACK_STATE" != "PENDING" ]]; then
      echo "A100 fallback left pending state before its hold took effect" >&2
      RACE_LOSER=1
      rm -f "$CLAIM_DIR/job_id"
      rmdir "$CLAIM_DIR"
      exit 1
    fi
  fi
  PRIMARY_STATE="$(squeue -h -j "$A100_PRIMARY" -o '%T' | head -n 1)"
  if [[ "$PRIMARY_STATE" == "RUNNING" || "$PRIMARY_STATE" == "COMPLETING" ]]; then
    echo "A100 primary started first; A40 rescue exits without opening output"
    RACE_LOSER=1
    rm -f "$CLAIM_DIR/job_id"
    rmdir "$CLAIM_DIR"
    exit 0
  fi
  if [[ "$PRIMARY_STATE" == "PENDING" ]]; then
    scancel "$A100_PRIMARY" "$A100_PRIMARY_AUDIT" || true
    for _ in $(seq 1 30); do
      [[ -z "$(squeue -h -j "$A100_PRIMARY")" ]] && break
      sleep 1
    done
    if [[ -n "$(squeue -h -j "$A100_PRIMARY")" ]]; then
      echo "A100 primary did not reach terminal state after cancellation" >&2
      exit 1
    fi
  fi
fi

copy_results() {
  if [[ -d "$OUTPUT" ]]; then
    mkdir -p "$HOME_COPY"
    rsync -a "$OUTPUT/" "$HOME_COPY/"
  fi
}

finish_or_continue() {
  status=$?
  copy_results
  if [[ "$RACE_LOSER" == "1" ]]; then
    return "$status"
  fi
  if [[ -f "$OUTPUT/generations.jsonl" && -f "$OUTPUT/summary.json" ]]; then
    scancel "$A100_FALLBACK" "$A100_FALLBACK_AUDIT" || true
    rm -f "$CLAIM_DIR/job_id"
    rmdir "$CLAIM_DIR"
    sbatch scripts/slurm/export_official_safety_audit.sbatch
  elif (( ATTEMPT < MAX_ATTEMPTS )); then
    next_attempt=$((ATTEMPT + 1))
    rm -f "$CLAIM_DIR/job_id"
    rmdir "$CLAIM_DIR"
    sbatch scripts/slurm/generate_official_coco_u3_k6_a40_rescue.sbatch "$next_attempt"
  else
    rm -f "$CLAIM_DIR/job_id"
    rmdir "$CLAIM_DIR"
    scontrol release "$A100_FALLBACK"
  fi
  return "$status"
}
trap finish_or_continue EXIT

if [[ ! -d "$OUTPUT" && -d "$HOME_COPY" ]]; then
  mkdir -p "$OUTPUT"
  rsync -a "$HOME_COPY/" "$OUTPUT/"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -u scripts/generate_official_safety.py \
  --config configs/official_safety_endpoints.yaml \
  --condition coco_u3_k6 \
  --output-dir "$OUTPUT"

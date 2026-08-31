#!/usr/bin/env bash
#SBATCH --job-name=mats-export-official-audit
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
cd "$CODE_ROOT"
python scripts/export_official_safety_audit.py \
  --config configs/official_safety_human_audit.yaml
python scripts/render_safety_audit_html.py \
  artifacts/discovery/results/official_safety/human_audit/audit_blinded.jsonl \
  artifacts/discovery/results/official_safety/human_audit/audit_blind_review.html
sha256sum \
  artifacts/discovery/results/official_safety/human_audit/audit_blinded.jsonl \
  artifacts/discovery/results/official_safety/human_audit/audit_blind_key.json \
  artifacts/discovery/results/official_safety/human_audit/audit_blind_review.html

#!/usr/bin/env bash
#SBATCH --job-name=mats-copy-coco-s3
#SBATCH --account=piray_1101
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
scripts/copy_compact_stage_results.sh fallback_4b_skip0 3

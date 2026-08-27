#!/bin/bash
#SBATCH --job-name=latent
#SBATCH --account=piray_1101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err

set -euo pipefail
PROJ=/scratch1/$USER/mats_latent_safety

module purge
module load gcc/13.3.0 cuda/12.6.3

export HF_HOME=$PROJ/cache/huggingface
export TOKENIZERS_PARALLELISM=false
# Verified 2026-08-21: compute nodes have outbound HTTPS, so HF downloads work
# in-job. Still prefer warming the cache on the login node for big weights —
# it avoids burning GPU-hours on a download.

source "$PROJ/.venv/bin/activate"
cd "$PROJ/code"

nvidia-smi
srun python -u "$@"

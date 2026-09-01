#!/usr/bin/env bash
#SBATCH --job-name=mats-finalize-safety-fig
#SBATCH --account=piray_1101
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.out
#SBATCH --error=/scratch1/muttreja/mats_latent_safety/logs/%x-%j.err
set -euo pipefail
source "/home1/$USER/mats_latent_safety/scripts/discovery_env.sh"
cd "$CODE_ROOT"
python -u scripts/build_dense_safety_trajectory.py
python -u scripts/plot_dense_safety_trajectory.py
sha256sum \
  artifacts/discovery/results/dense_safety/trajectory/trajectory.json \
  artifacts/discovery/results/dense_safety/trajectory/trajectory.csv \
  artifacts/discovery/results/dense_safety/trajectory/trajectory.md \
  artifacts/discovery/results/dense_safety/trajectory/trajectory.png \
  artifacts/discovery/results/dense_safety/trajectory/trajectory.svg

#!/usr/bin/env bash
# One-time environment setup on Discovery. Run on the login node (it has internet).
#   ssh discovery 'bash /scratch1/muttreja/mats_latent_safety/code/scripts/bootstrap_env.sh'
#
# Everything lands on /scratch1 — /home1 is at 68/100 GiB and a torch env plus
# model weights would fill it.
set -euo pipefail

PROJ=/scratch1/$USER/mats_latent_safety

module purge
module load gcc/13.3.0 cuda/12.6.3

export UV_CACHE_DIR=$PROJ/cache/uv
export HF_HOME=$PROJ/cache/huggingface

uv venv --python 3.11 "$PROJ/.venv"
source "$PROJ/.venv/bin/activate"

# cu126 wheels to match the loaded CUDA module.
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers datasets accelerate einops numpy scipy pandas \
  matplotlib seaborn tqdm wandb jaxtyping

echo "env ready: $PROJ/.venv"
python -c "import torch; print('torch', torch.__version__)"

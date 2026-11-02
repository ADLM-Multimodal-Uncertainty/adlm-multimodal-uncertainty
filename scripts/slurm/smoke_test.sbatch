#!/bin/bash
#SBATCH --job-name=adlm-smoke
# Logs go to miltank, never to the home dir (submit dir may be ~). Directory must exist before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# First job for every team member: proves VPN, ssh, storage, uv env and GPU allocation work.
set -euo pipefail
ml python/uv
export UV_CACHE_DIR="${UV_CACHE_DIR:-/meta/users/$USER/.cache/uv}"  # keep uv cache out of ~
cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
nvidia-smi
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run pytest -q

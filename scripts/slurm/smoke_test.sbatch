#!/bin/bash
#SBATCH --job-name=adlm-smoke
# Logs go to miltank, never to the home directory. Create it once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# First job on a new checkout: GPU visibility plus the synthetic test suite.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/slurm/_common.sh"
adlm_prepare_caches
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PYTHON" -m pytest -q

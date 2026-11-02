#!/bin/bash
#SBATCH --job-name=adlm-train
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Usage: sbatch scripts/slurm/train.sbatch <hydra overrides>
set -euo pipefail
ml python/uv
cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/$USER/outputs}"
echo "commit $(git rev-parse --short HEAD)"
uv run python -m src.train "$@"

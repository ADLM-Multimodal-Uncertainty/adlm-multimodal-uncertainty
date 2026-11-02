#!/bin/bash
#SBATCH --job-name=adlm-ens
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-24%2

# 5 folds x 5 seeds = 25 tasks, 2 at a time (master-queuesave limit).
# Usage: sbatch scripts/slurm/ensemble_array.sbatch <hydra overrides>
set -euo pipefail
ml python/uv
cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/$USER/outputs}"
FOLD=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
echo "commit $(git rev-parse --short HEAD) fold ${FOLD} seed ${SEED}"
uv run python -m src.train fold="${FOLD}" seed="${SEED}" "$@"

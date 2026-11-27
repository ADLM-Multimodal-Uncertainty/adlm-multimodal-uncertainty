#!/bin/bash
#SBATCH --job-name=adlm-train
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

# One training job. Pass Hydra overrides after the script name, including fold= and experiment=.
#   sbatch scripts/slurm/train.sbatch experiment=baseline_image_only fold=0 '~ensemble'
set -euo pipefail
REPO_DIR="${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/slurm/_common.sh"
adlm_prepare_caches
adlm_resolve_data
adlm_stage_data

echo "commit $(git rev-parse --short HEAD) host=$(hostname) data=${RESOLVED_DATA} local=${ADLM_LOCAL_DATA_ROOT:-none}"
"$PYTHON" -m src.train \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

#!/bin/bash
#SBATCH --job-name=adlm-fold
# Logs go to miltank, never to ~. Create the directory once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-1%1

# One array task trains one fold of a deep ensemble (five members, sequentially).
# %1 keeps one task of this array running. master-queuesave allows two jobs per user,
# so a paired image-only plus multimodal submission stays at exactly two jobs.
# Do not raise the throttle to %2 while another queuesave array is pending or running.
#
#   sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=baseline_image_only
#   sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=baseline_multimodal
set -euo pipefail
REPO_DIR="${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/slurm/_common.sh"
adlm_prepare_caches
adlm_resolve_data
adlm_stage_data

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "commit $(git rev-parse --short HEAD) host=$(hostname) fold=${FOLD} data=${RESOLVED_DATA} local=${ADLM_LOCAL_DATA_ROOT:-none}"
"$PYTHON" -m src.train \
    fold="${FOLD}" \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

#!/bin/bash
#SBATCH --job-name=adlm-eu-fold
# Logs go to miltank, never to ~. Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-02:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-4%2

# Fold-level deep-ensemble array: each array task trains one complete 5-member DeepEnsemble fold.
# %2 respects master-queuesave's limit of at most 2 concurrent non-interruptible jobs per user.
#
# Examples:
#   sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=eu_baseline_r18_image_only
#   sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_multimodal
#   sbatch --array=2-4%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_multimodal
set -euo pipefail
ml python/uv

export UV_CACHE_DIR="${UV_CACHE_DIR:-/meta/users/$USER/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/meta/users/$USER/cache}"
export TORCH_HOME="${TORCH_HOME:-/meta/users/$USER/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/meta/users/$USER/cache/matplotlib}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/$USER/outputs}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$OUTPUT_ROOT/slurm"

cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"

resolve_data_dir() {
    for candidate in \
        "${DATA_DIR:-}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad_ufes_20}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad-ufes-20}" \
        "/vol/miltank/users/$USER/data/pad_ufes_20"; do
        if [[ -n "$candidate" && -f "$candidate/metadata.csv" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

RESOLVED_DATA="$(resolve_data_dir)" || {
    echo "ERROR: PAD-UFES-20 metadata.csv not found. Export DATA_DIR or DATA_ROOT on /vol/miltank." >&2
    exit 1
}

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "commit $(git rev-parse --short HEAD) fold=${FOLD} data=${RESOLVED_DATA} output_root=${OUTPUT_ROOT}"
uv run python -m src.train \
    fold="${FOLD}" \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

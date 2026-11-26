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
#   sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=eu_v2_r18_image_only
#   sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_v2_r18_multimodal
#   sbatch --array=2-4%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_v2_r18_multimodal
set -euo pipefail

CACHE_BASE="/meta/users/$USER/.cache"
if [[ ! -d "$CACHE_BASE" || ! -w "$CACHE_BASE" ]]; then
    CACHE_BASE="/vol/miltank/users/$USER/.cache"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_BASE/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_BASE}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_BASE/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-/vol/miltank/users/$USER/checkpoints/torch}"
export HF_HOME="${HF_HOME:-/vol/miltank/users/$USER/checkpoints/huggingface}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/$USER/outputs}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" "$OUTPUT_ROOT/slurm"

cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found. Run 'ml python/uv && uv sync' in $PWD first." >&2
    exit 1
fi

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

# Stage dataset once onto node-local NVMe (/tmp) to avoid repeated NFS PNG reads and EIO stalls.
STAGE_DIR=""
cleanup_stage() {
    if [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]]; then
        rm -rf "$STAGE_DIR"
    fi
}
trap cleanup_stage EXIT

if [[ "${STAGE_TO_LOCAL:-1}" == "1" && -w /tmp ]]; then
    STAGE_DIR="/tmp/adlm_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$STAGE_DIR/pad_ufes_20"
    DATA_TAR="${RESOLVED_DATA}.tar"
    if [[ -f "$DATA_TAR" ]]; then
        echo "staging $DATA_TAR -> $STAGE_DIR/pad_ufes_20"
        tar -xf "$DATA_TAR" -C "$STAGE_DIR/pad_ufes_20"
    else
        echo "staging $RESOLVED_DATA -> $STAGE_DIR/pad_ufes_20"
        cp -a "$RESOLVED_DATA/." "$STAGE_DIR/pad_ufes_20/"
    fi
    export ADLM_LOCAL_DATA_ROOT="$STAGE_DIR/pad_ufes_20"
fi

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "commit $(git rev-parse --short HEAD) host=$(hostname) fold=${FOLD} data=${RESOLVED_DATA} local=${ADLM_LOCAL_DATA_ROOT:-none} output_root=${OUTPUT_ROOT}"
"$PYTHON" -m src.train \
    fold="${FOLD}" \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

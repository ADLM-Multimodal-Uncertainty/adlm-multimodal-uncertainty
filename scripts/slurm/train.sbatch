#!/bin/bash
#SBATCH --job-name=adlm-train
# Logs go to miltank, never to ~. Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

# Usage: sbatch scripts/slurm/train.sbatch experiment=eu_baseline_r18_multimodal fold=0
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

echo "commit $(git rev-parse --short HEAD) host=$(hostname) data=$RESOLVED_DATA output_root=$OUTPUT_ROOT"
"$PYTHON" -m src.train \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

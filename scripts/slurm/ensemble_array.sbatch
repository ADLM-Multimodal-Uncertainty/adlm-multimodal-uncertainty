#!/bin/bash
#SBATCH --job-name=adlm-ens
# Logs go to miltank, never to ~. Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A_%a.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-24%2

# 5 folds x 5 standalone seeds = 25 single-model tasks, 2 at a time (master-queuesave limit).
# Use ONLY for single-model presets (ensemble=null). For DeepEnsemble presets (eu_*, deep_ensemble*),
# use scripts/slurm/fold_ensemble_array.sbatch instead so you do not launch 25x5=125 models.
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        experiment=eu_*|experiment=deep_ensemble*|ensemble=deep)
            echo "ERROR: $arg already uses DeepEnsemble. Submit scripts/slurm/fold_ensemble_array.sbatch instead." >&2
            exit 1
            ;;
    esac
done

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

FOLD=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
echo "commit $(git rev-parse --short HEAD) host=$(hostname) fold=${FOLD} seed=${SEED} data=${RESOLVED_DATA} local=${ADLM_LOCAL_DATA_ROOT:-none}"
"$PYTHON" -m src.train \
    fold="${FOLD}" \
    seed="${SEED}" \
    output_root="$OUTPUT_ROOT" \
    data.root="$RESOLVED_DATA" \
    data.num_workers="${NUM_WORKERS:-4}" \
    device=cuda \
    "$@"

#!/bin/bash
#SBATCH --job-name=adlm-smoke
# Logs go to miltank, never to the home dir (submit dir may be ~). Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# First job for every team member: proves VPN, ssh, miltank/meta storage, uv env, GPU,
# tests, and (when PAD-UFES-20 is present) a 1-epoch CUDA smoke run all work.
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
PYTEST="${PYTEST:-.venv/bin/pytest}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found. Run 'ml python/uv && uv sync --extra dev' in $PWD first." >&2
    exit 1
fi
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current) host $(hostname)"
echo "caches: UV=$UV_CACHE_DIR TORCH=$TORCH_HOME MPL=$MPLCONFIGDIR"
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
"$PYTEST" -q

resolve_data_dir() {
    for candidate in \
        "${DATA_DIR:-}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad_ufes_20}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad-ufes-20}" \
        "/vol/miltank/users/$USER/data/pad_ufes_20" \
        "data/pad_ufes_20"; do
        if [[ -n "$candidate" && -f "$candidate/metadata.csv" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if RESOLVED_DATA="$(resolve_data_dir)"; then
    echo "running 1-epoch CUDA smoke training with data.root=$RESOLVED_DATA"
    "$PYTHON" -m src.train \
        experiment=eu_v2_r18_multimodal \
        '~ensemble' \
        name=cluster_smoke \
        output_root="$OUTPUT_ROOT/smoke" \
        data.root="$RESOLVED_DATA" \
        data.subset=64 \
        data.image_size=64 \
        data.num_workers=2 \
        train.epochs=1 \
        model.image_encoder.pretrained=false \
        device=cuda
else
    echo "PAD-UFES-20 not found yet; set DATA_DIR or DATA_ROOT before training runs."
fi

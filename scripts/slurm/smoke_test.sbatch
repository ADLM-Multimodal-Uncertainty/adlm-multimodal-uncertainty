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
ml python/uv

export UV_CACHE_DIR="${UV_CACHE_DIR:-/meta/users/$USER/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/meta/users/$USER/cache}"
export TORCH_HOME="${TORCH_HOME:-/meta/users/$USER/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/meta/users/$USER/cache/matplotlib}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/$USER/outputs}"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$OUTPUT_ROOT/slurm"

cd "${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"
echo "caches: UV=$UV_CACHE_DIR TORCH=$TORCH_HOME MPL=$MPLCONFIGDIR"
nvidia-smi
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run pytest -q

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
    uv run python -m src.train \
        experiment=eu_baseline_r18_multimodal \
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

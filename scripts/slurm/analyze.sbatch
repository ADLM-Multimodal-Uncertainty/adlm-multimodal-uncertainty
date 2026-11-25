#!/bin/bash
#SBATCH --job-name=adlm-analyze
# CPU-only job: never request a GPU for post-hoc EU analysis and plotting!
# Logs go to miltank, never to ~. Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage:
#   sbatch scripts/slurm/analyze.sbatch
#   sbatch scripts/slurm/analyze.sbatch analysis.output_dir=/vol/miltank/users/$USER/outputs/analysis_stage2
#   sbatch scripts/slurm/analyze.sbatch analysis.manifest=/vol/miltank/users/$USER/outputs/analysis/manifest.json
set -eo pipefail
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
ml python/uv
set -u

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
echo "commit $(git rev-parse --short HEAD) output_root=$OUTPUT_ROOT"

uv run --no-sync python -m src.analyze \
    analysis.runs_root="$OUTPUT_ROOT/runs" \
    analysis.output_dir="$OUTPUT_ROOT/analysis" \
    "$@"

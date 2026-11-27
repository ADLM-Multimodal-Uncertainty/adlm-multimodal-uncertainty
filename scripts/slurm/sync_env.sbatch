#!/bin/bash
#SBATCH --job-name=adlm-sync-env
# Logs go to miltank, never to ~. Create once before sbatch:
#   mkdir -p /vol/miltank/users/$USER/outputs/slurm
#SBATCH --output=/vol/miltank/users/%u/outputs/slurm/%x-%A.out
#SBATCH --error=/vol/miltank/users/%u/outputs/slurm/%x-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU-only. Installs the shared Python and project packages. No GPU: this job only downloads.
# Submit from selene; do not run uv python install or uv sync on the login node.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty}"
cd "$REPO_DIR"

if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    ml python/uv
fi

CACHE_BASE="/meta/users/${USER}/.cache"
if [[ ! -d "$CACHE_BASE" || ! -w "$CACHE_BASE" ]]; then
    CACHE_BASE="/vol/miltank/users/${USER}/.cache"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_BASE/uv}"
export UV_PYTHON_INSTALL_DIR="/vol/miltank/users/${USER}/python"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "/vol/miltank/users/${USER}/outputs/slurm"

uv python install 3.12
PY="$(uv python find 3.12)"
case "$PY" in
    /vol/miltank/*) ;;
    *)
        echo "ERROR: uv python find returned $PY. Refusing a node-local /usr/bin interpreter." >&2
        exit 1
        ;;
esac

rm -rf .venv
uv venv --python "$PY"
uv sync --extra dev
.venv/bin/python -c "import sys, hydra; print(sys.executable); print(sys.version); print('hydra', hydra.__version__)"

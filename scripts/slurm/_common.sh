#!/bin/bash
# Shared cluster setup for the SLURM scripts. Source it from the repository copy:
# SLURM may spool the submitted script away from scripts/slurm, so do not use dirname "$0".

adlm_prepare_caches() {
    local cache_base="/meta/users/${USER}/.cache"
    if [[ ! -d "$cache_base" || ! -w "$cache_base" ]]; then
        cache_base="/vol/miltank/users/${USER}/.cache"
    fi
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_base/uv}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_base}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_base/matplotlib}"
    export TORCH_HOME="${TORCH_HOME:-/vol/miltank/users/${USER}/checkpoints/torch}"
    export HF_HOME="${HF_HOME:-/vol/miltank/users/${USER}/checkpoints/huggingface}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/vol/miltank/users/${USER}/outputs}"
    mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" \
        "$OUTPUT_ROOT/slurm" "$OUTPUT_ROOT/runs"

    cd "${REPO_DIR:-/vol/miltank/users/${USER}/code/adlm-multimodal-uncertainty}"
    # Compute nodes often export PYTHON to the system interpreter, which has no project
    # dependencies. Always use the checkout's virtualenv.
    PYTHON="${PWD}/.venv/bin/python"
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: $PYTHON not found. On the login node run: ml python/uv && uv sync" >&2
        return 1
    fi
    export PYTHON
}

adlm_resolve_data() {
    local candidate
    RESOLVED_DATA=""
    for candidate in \
        "${DATA_DIR:-}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad_ufes_20}" \
        "${DATA_ROOT:+${DATA_ROOT}/pad-ufes-20}" \
        "/vol/miltank/users/${USER}/data/pad_ufes_20"; do
        if [[ -n "$candidate" && -f "$candidate/metadata.csv" ]]; then
            RESOLVED_DATA="$candidate"
            break
        fi
    done
    if [[ -z "$RESOLVED_DATA" ]]; then
        echo "ERROR: PAD-UFES-20 metadata.csv not found. Export DATA_DIR or DATA_ROOT." >&2
        return 1
    fi
    export RESOLVED_DATA
}

adlm_stage_data() {
    STAGE_DIR=""
    cleanup_stage() {
        if [[ -n "${STAGE_DIR:-}" && -d "$STAGE_DIR" ]]; then
            rm -rf "$STAGE_DIR"
        fi
    }
    trap cleanup_stage EXIT
    if [[ "${STAGE_TO_LOCAL:-1}" == "1" && -w /tmp ]]; then
        STAGE_DIR="/tmp/adlm_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
        mkdir -p "$STAGE_DIR/pad_ufes_20"
        if [[ -f "${RESOLVED_DATA}.tar" ]]; then
            echo "staging ${RESOLVED_DATA}.tar -> $STAGE_DIR/pad_ufes_20"
            tar -xf "${RESOLVED_DATA}.tar" -C "$STAGE_DIR/pad_ufes_20"
        else
            echo "staging $RESOLVED_DATA -> $STAGE_DIR/pad_ufes_20"
            cp -a "$RESOLVED_DATA/." "$STAGE_DIR/pad_ufes_20/"
        fi
        export ADLM_LOCAL_DATA_ROOT="$STAGE_DIR/pad_ufes_20"
    fi
}

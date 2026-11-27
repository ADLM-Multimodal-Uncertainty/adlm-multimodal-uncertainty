#!/usr/bin/env bash
# Copy finished cluster runs into the local outputs/runs directory.
# Checkpoints are skipped unless INCLUDE_CHECKPOINTS=1, because the report reads
# predictions, histories, and configs rather than the weight files.
#
#   scripts/local/sync_cluster_runs.sh baseline_image_only baseline_multimodal
#   CLUSTER_USER=kalem scripts/local/sync_cluster_runs.sh baseline_image_only
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $# -lt 1 ]]; then
    echo "Usage: scripts/local/sync_cluster_runs.sh <run_group> [more run groups]" >&2
    exit 1
fi

CLUSTER_HOST="${CLUSTER_HOST:-selene}"
CLUSTER_USER="${CLUSTER_USER:-$USER}"
REMOTE_ROOT="${CLUSTER_OUTPUT_ROOT:-/vol/miltank/users/${CLUSTER_USER}/outputs/runs}"
LOCAL_ROOT="${LOCAL_RUNS_ROOT:-outputs/runs}"
excludes=()
if [[ "${INCLUDE_CHECKPOINTS:-0}" != "1" ]]; then
    excludes=(--exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt')
fi

mkdir -p "$LOCAL_ROOT"
for group in "$@"; do
    mkdir -p "$LOCAL_ROOT/$group"
    echo "sync ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_ROOT}/${group}/ -> ${LOCAL_ROOT}/${group}/"
    rsync -a "${excludes[@]}" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_ROOT}/${group}/" \
        "${LOCAL_ROOT}/${group}/"
done

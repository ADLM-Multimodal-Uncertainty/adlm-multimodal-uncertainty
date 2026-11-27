#!/usr/bin/env bash
# One-epoch run on a few small images with a randomly initialised backbone (no weight
# download). Checks the whole pipeline; the predictions are not a result.
#   scripts/local/smoke.sh experiment=baseline_multimodal
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.train \
    output_root=outputs/smoke \
    data.subset=64 \
    data.image_size=64 \
    train.epochs=1 \
    train.weighted_loss=false \
    model.image_encoder.pretrained=false \
    model.image_encoder.freeze=false \
    "$@" \
    '~ensemble'

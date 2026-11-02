#!/usr/bin/env bash
# One-epoch run on a few small images with a randomly initialised backbone (no weight
# download). Checks the whole pipeline in about a minute; results are meaningless.
#   scripts/local/smoke.sh experiment=draft_image_only
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.train output_root=outputs/smoke data.subset=64 data.image_size=64 \
    train.epochs=1 model.image_encoder.pretrained=false model.image_encoder.freeze=false "$@"

#!/usr/bin/env bash
# Train and evaluate one run locally. Arguments are Hydra overrides, e.g.
#   scripts/local/run.sh experiment=draft_baseline
#   scripts/local/run.sh experiment=draft_image_only data.root=/path/to/pad_ufes_20
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.train "$@"

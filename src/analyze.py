"""Write the standardized tables, figures, and report for one experiment.

Single setup, image-only versus image plus metadata:

    uv run python -m src.analyze \
        experiment_id=default_baseline eval_split=test \
        reference=baseline_image_only candidate=baseline_multimodal

Baseline recipe versus a candidate recipe, each with both setups:

    uv run python -m src.analyze \
        experiment_id=aug_crop_vs_baseline eval_split=val \
        baseline_image=baseline_image_only \
        baseline_multimodal=baseline_multimodal \
        candidate_image=crop_zoom_image_only \
        candidate_multimodal=crop_zoom_multimodal \
        candidate_label="Crop and zoom"
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from src.analysis import run_analysis

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="analyze", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output = run_analysis(cfg)
    log.info("wrote %s", output)
    print(output)


if __name__ == "__main__":
    main()

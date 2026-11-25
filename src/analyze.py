"""Analyze paired image-only and multimodal deep-ensemble artifacts.

    python -m src.analyze analysis.manifest=outputs/analysis/manifest.json
"""

from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="analyze", version_base="1.3")
def main(cfg: DictConfig) -> None:
    instantiate(cfg.analysis).run()


if __name__ == "__main__":
    main()

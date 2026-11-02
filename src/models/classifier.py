"""Image encoder + optional tabular encoder + fusion head, assembled from the `model` config."""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn


class MultimodalClassifier(nn.Module):
    def __init__(
        self, image_encoder: nn.Module, tabular_encoder: nn.Module | None, fusion: nn.Module
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion = fusion

    @property
    def uses_metadata(self) -> bool:
        return self.tabular_encoder is not None

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        image_feats = self.image_encoder(image)
        meta_feats = None
        if self.tabular_encoder is not None:
            if meta is None:
                raise ValueError("this model has a metadata branch and needs `meta`")
            meta_feats = self.tabular_encoder(meta)
        return self.fusion(image_feats, meta_feats)


def build_model(cfg: DictConfig, n_classes: int, n_meta_features: int) -> MultimodalClassifier:
    """Instantiate the `_target_` of each component in the `model` config.

    Sizes that depend on the data or on another component are passed here, never set in YAML.
    """
    image_encoder = instantiate(cfg.image_encoder)
    tabular_encoder = None
    if cfg.get("tabular_encoder") is not None:
        tabular_encoder = instantiate(cfg.tabular_encoder, in_dim=n_meta_features)
    fusion = instantiate(
        cfg.fusion,
        image_dim=image_encoder.out_dim,
        meta_dim=tabular_encoder.out_dim if tabular_encoder is not None else 0,
        n_classes=n_classes,
    )
    return MultimodalClassifier(image_encoder, tabular_encoder, fusion)

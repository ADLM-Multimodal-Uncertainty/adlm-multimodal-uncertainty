"""Fusion heads, selected by `configs/model/fusion/<name>.yaml`.

Contract: `__init__(image_dim, meta_dim, n_classes, ...)` (sizes passed by `build_model`) and
`forward(image_feats (B, image_dim), meta_feats (B, meta_dim) | None) -> logits (B, n_classes)`.
`meta_dim == 0` and `meta_feats is None` mean the model has no metadata branch (image-only).
"""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Concatenate both feature vectors, then an MLP classifier."""

    def __init__(
        self,
        image_dim: int,
        meta_dim: int,
        n_classes: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(image_dim + meta_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image_feats: torch.Tensor, meta_feats: torch.Tensor | None) -> torch.Tensor:
        if meta_feats is not None:
            image_feats = torch.cat([image_feats, meta_feats], dim=1)
        return self.head(image_feats)

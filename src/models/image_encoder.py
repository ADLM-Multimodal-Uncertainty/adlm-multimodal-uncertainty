"""Image encoders, selected by `configs/model/image_encoder/<name>.yaml`.

Contract: `forward(image (B, 3, H, W)) -> features (B, out_dim)`, with `out_dim` set in
`__init__` because `build_model` sizes the fusion head from it.
"""

from __future__ import annotations

import torch
import torchvision
from torch import nn


class TorchvisionResNetEncoder(nn.Module):
    """torchvision ResNet without its classifier, followed by an optional projection head."""

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = True,
        proj_dim: int | None = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if freeze and not pretrained:
            raise ValueError("cannot freeze a randomly initialised backbone")
        self.backbone = getattr(torchvision.models, arch)(weights="DEFAULT" if pretrained else None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.freeze = freeze
        if freeze:
            self.backbone.requires_grad_(False)
        if proj_dim:
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
        self.out_dim: int = proj_dim or feat_dim

    def train(self, mode: bool = True) -> TorchvisionResNetEncoder:
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            feats = self.backbone(image)
        return self.projection(feats)

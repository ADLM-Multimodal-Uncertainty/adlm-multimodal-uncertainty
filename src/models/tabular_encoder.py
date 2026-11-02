"""Metadata encoders, selected by `configs/model/tabular_encoder/<name>.yaml`.

Contract: `__init__(in_dim, ...)` where `in_dim` is the encoded metadata width (passed by
`build_model`), and `forward(meta (B, in_dim)) -> features (B, out_dim)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPTabularEncoder(nn.Module):
    """Linear -> ReLU -> Dropout per hidden layer."""

    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int] = (64,), dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for hidden in hidden_dims:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim: int = in_dim

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)

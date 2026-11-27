"""Training losses selectable through `train.loss._target_`.

Contract: `__init__` accepts `weight` (per-class tensor or None, set by `train.weighted_loss`);
`forward(logits, target)` returns the scalar mean loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017): -w_y (1 - p_y)^gamma log p_y.

    Averaged like `nn.CrossEntropyLoss`: a plain mean without `weight`, otherwise divided by
    the summed weights of the batch. `gamma=0` reproduces cross-entropy exactly.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - log_p.exp()).clamp(min=0) ** self.gamma) * log_p
        if self.weight is None:
            return loss.mean()
        w = self.weight[target]
        return (w * loss).sum() / w.sum()

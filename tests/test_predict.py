import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluate import predict


class FixedLogitsModel(nn.Module):
    """Returns precomputed logits for the batch, keyed by the image tensor's first value."""

    uses_metadata = False

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, image, meta=None):
        return self.logits[image[:, 0, 0, 0].long()]


def _loader(n: int, labels: torch.Tensor, batch_size: int) -> DataLoader:
    images = torch.arange(n).float().view(n, 1, 1, 1).expand(n, 3, 1, 1)
    items = [{"image": images[i], "label": labels[i]} for i in range(n)]
    return DataLoader(items, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [2, 7])
def test_loss_is_the_mean_over_all_samples_regardless_of_batch_size(batch_size):
    torch.manual_seed(0)
    logits, labels = torch.randn(7, 4), torch.randint(0, 4, (7,))
    criterion = nn.CrossEntropyLoss()
    result = predict(FixedLogitsModel(logits), _loader(7, labels, batch_size), "cpu", criterion)
    assert result.loss == pytest.approx(criterion(logits, labels).item(), rel=1e-6)
    np.testing.assert_allclose(result.probs, torch.softmax(logits, dim=1).numpy(), rtol=1e-6)
    np.testing.assert_array_equal(result.labels, labels.numpy())


def test_no_criterion_means_no_loss():
    logits, labels = torch.randn(3, 4), torch.zeros(3, dtype=torch.long)
    assert predict(FixedLogitsModel(logits), _loader(3, labels, 3), "cpu").loss is None

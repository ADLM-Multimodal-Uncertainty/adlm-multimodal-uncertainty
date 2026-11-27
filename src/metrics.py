"""Classification and proper scoring metrics.

Classification fractions are in ``[0, 1]``. Reports convert them to percentages.
Negative log-likelihood is in nats. The multiclass Brier score sums the squared
error over classes, so it lies in ``[0, 2]``.

Macro-F1 and balanced accuracy average over classes that have at least one true
image in the evaluated set. A class that a fold does not contain is omitted rather
than counted as a zero. Per-class precision, recall, and F1 still use
``zero_division=0`` when a present class has an empty predicted or true count.
The confusion matrix always keeps the declared class order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")
PROBABILITY_FLOOR = 1e-12


def summary_metrics(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> dict[str, float]:
    """The metrics in `METRIC_NAMES`."""
    labels = np.asarray(labels)
    present = [index for index in range(n_classes) if np.any(labels == index)]
    macro_labels = present or list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(
                labels, preds, labels=macro_labels, average="macro", zero_division=0
            )
        ),
    }


def detailed_metrics(labels: np.ndarray, preds: np.ndarray, classes: Sequence[str]) -> dict:
    """Summary metrics plus the per-class report and the confusion matrix (rows = truth)."""
    ids = list(range(len(classes)))
    return {
        **summary_metrics(labels, preds, len(classes)),
        "classification_report": classification_report(
            labels, preds, labels=ids, target_names=list(classes), zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(labels, preds, labels=ids).tolist(),
        "class_order": list(classes),
    }


def nll(labels: np.ndarray, probs: np.ndarray, floor: float = PROBABILITY_FLOOR) -> float:
    """Mean ``-log q(y)`` in nats. A probability of 1 stays exact 0; only the floor clips zeros."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    chosen = probs[np.arange(len(labels)), labels]
    chosen = np.maximum(chosen, floor)
    return float(-np.log(chosen).mean())


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean over images of the sum of squared errors across classes."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def probability_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Unweighted negative log-likelihood and multiclass Brier score."""
    return {"nll": nll(labels, probs), "brier": brier_score(labels, probs)}

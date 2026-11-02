"""Classification metrics on predicted class indices."""

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


def summary_metrics(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(labels, preds, labels=range(n_classes), average="macro", zero_division=0)
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

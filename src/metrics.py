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
    log_loss,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")


def summary_metrics(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> dict[str, float]:
    """The metrics in `METRIC_NAMES`."""
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(
            f1_score(labels, preds, labels=range(n_classes), average="macro", zero_division=0)
        ),
    }


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE with equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            include |= confidence == 0
        if include.any():
            error += include.mean() * abs(correct[include].mean() - confidence[include].mean())
    return float(error)


def probability_metrics(
    labels: np.ndarray, probs: np.ndarray, n_classes: int
) -> dict[str, float | None]:
    """Proper scoring and calibration metrics for multiclass probabilities."""
    expected_shape = (len(labels), n_classes)
    if probs.shape != expected_shape:
        raise ValueError(f"expected probabilities with shape {expected_shape}, got {probs.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    probs = probs / row_sums[:, None]

    one_hot = np.eye(n_classes)[labels]
    present = np.unique(labels)
    macro_auroc = None
    if len(present) == n_classes:
        if n_classes == 2:
            macro_auroc = float(roc_auc_score(labels, probs[:, 1]))
        else:
            macro_auroc = float(
                roc_auc_score(
                    labels,
                    probs,
                    labels=list(range(n_classes)),
                    average="macro",
                    multi_class="ovr",
                )
            )
    return {
        "negative_log_likelihood": float(
            log_loss(labels, probs, labels=list(range(n_classes)))
        ),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(labels, probs),
        "macro_ovr_auroc": macro_auroc,
    }


def detailed_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    classes: Sequence[str],
    probs: np.ndarray | None = None,
) -> dict:
    """Summary metrics plus the per-class report and the confusion matrix (rows = truth)."""
    ids = list(range(len(classes)))
    report = classification_report(
        labels, preds, labels=ids, target_names=list(classes), zero_division=0, output_dict=True
    )
    confusion = confusion_matrix(labels, preds, labels=ids)
    specificity = {}
    for index, name in enumerate(classes):
        true_negative = confusion.sum() - confusion[index].sum() - confusion[:, index].sum()
        true_negative += confusion[index, index]
        false_positive = confusion[:, index].sum() - confusion[index, index]
        denominator = true_negative + false_positive
        specificity[name] = float(true_negative / denominator) if denominator else 0.0

    metrics = {
        **summary_metrics(labels, preds, len(classes)),
        "weighted_f1": float(
            f1_score(labels, preds, labels=ids, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "specificity": specificity,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(classes),
    }
    if "MEL" in report:
        metrics["mel_recall"] = float(report["MEL"]["recall"])
        metrics["mel_specificity"] = specificity["MEL"]
    if probs is not None:
        metrics.update(probability_metrics(labels, probs, len(classes)))
        per_class_auroc: dict[str, float | None] = {}
        for index, name in enumerate(classes):
            binary = (labels == index).astype(int)
            per_class_auroc[name] = (
                float(roc_auc_score(binary, probs[:, index]))
                if len(np.unique(binary)) == 2
                else None
            )
        metrics["per_class_ovr_auroc"] = per_class_auroc
        if "MEL" in per_class_auroc:
            metrics["mel_ovr_auroc"] = per_class_auroc["MEL"]
        if "SCC" in per_class_auroc:
            metrics["scc_ovr_auroc"] = per_class_auroc["SCC"]
    return metrics

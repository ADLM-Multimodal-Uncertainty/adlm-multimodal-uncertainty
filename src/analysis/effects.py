"""Paired uncertainty reductions and correctness groups.

Reductions are reference minus candidate, so a positive value means the candidate is
less uncertain. Relative reduction is unavailable when the reference mean is 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.metrics import detailed_metrics, probability_metrics, summary_metrics


def reduction_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | str]:
    delta = np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    denominator = float(np.mean(reference))
    relative: float | str
    if denominator == 0:
        relative = "unavailable"
    else:
        relative = 100.0 * float(delta.mean()) / denominator
    return {
        "mean_nats": float(delta.mean()),
        "median_nats": float(np.median(delta)),
        "relative_pct": relative,
        "share_reduced_pct": 100.0 * float(np.mean(delta > 0)),
    }


def patient_cluster_mean_ci(
    patient_ids: np.ndarray, values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, float]:
    """Image-weighted mean, with a 95% interval from resampling whole patients."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if samples <= 0 or len(values) == 0:
        return mean, mean, mean
    patient_ids = np.asarray(patient_ids)
    groups = [values[patient_ids == patient] for patient in pd.unique(patient_ids)]
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        draws[index] = np.concatenate([groups[i] for i in chosen]).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def quadrant_counts(
    reference_correct: np.ndarray, candidate_correct: np.ndarray
) -> list[tuple[str, int]]:
    """Four correctness groups. Their counts sum to the number of paired images."""
    reference_correct = np.asarray(reference_correct, dtype=bool)
    candidate_correct = np.asarray(candidate_correct, dtype=bool)
    groups = (
        ("both_correct", reference_correct & candidate_correct),
        ("reference_wrong_candidate_correct", ~reference_correct & candidate_correct),
        ("reference_correct_candidate_wrong", reference_correct & ~candidate_correct),
        ("both_wrong", ~reference_correct & ~candidate_correct),
    )
    counts = [(name, int(mask.sum())) for name, mask in groups]
    if sum(count for _, count in counts) != len(reference_correct):
        raise RuntimeError("correctness groups do not cover every paired image")
    return counts


def performance_row(labels: np.ndarray, probs: np.ndarray, total, aleatoric, epistemic) -> dict:
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    summary = summary_metrics(labels, probs.argmax(axis=1), probs.shape[1])
    scoring = probability_metrics(labels, probs)
    return {
        "n_images": int(len(labels)),
        "accuracy_pct": 100.0 * summary["accuracy"],
        "balanced_accuracy_pct": 100.0 * summary["balanced_accuracy"],
        "macro_f1_pct": 100.0 * summary["macro_f1"],
        "nll_nats": scoring["nll"],
        "brier": scoring["brier"],
        "mean_total_uncertainty_nats": float(np.mean(total)),
        "mean_aleatoric_uncertainty_nats": float(np.mean(aleatoric)),
        "mean_epistemic_uncertainty_nats": float(np.mean(epistemic)),
    }


def class_scores(labels: np.ndarray, preds: np.ndarray, classes: Sequence[str]) -> pd.DataFrame:
    """Per-class precision, recall, and F1 in percent, including classes with no support."""
    report = detailed_metrics(labels, preds, classes)["classification_report"]
    rows = []
    for name in classes:
        entry = report[name]
        rows.append(
            {
                "diagnosis": name,
                "support": int(entry["support"]),
                "precision_pct": 100.0 * float(entry["precision"]),
                "recall_pct": 100.0 * float(entry["recall"]),
                "f1_pct": 100.0 * float(entry["f1-score"]),
            }
        )
    return pd.DataFrame(rows)

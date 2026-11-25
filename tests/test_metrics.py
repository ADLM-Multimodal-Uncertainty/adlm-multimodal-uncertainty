import numpy as np
import pytest

from src.metrics import detailed_metrics, expected_calibration_error, probability_metrics


def test_probability_metrics_for_confident_correct_predictions():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = probability_metrics(labels, probs, n_classes=2)

    assert metrics["macro_ovr_auroc"] == pytest.approx(1.0)
    assert metrics["negative_log_likelihood"] == pytest.approx(
        -np.log([0.9, 0.9, 0.8, 0.8]).mean()
    )
    assert metrics["multiclass_brier"] == pytest.approx(0.05)
    assert metrics["ece"] == pytest.approx(0.15)


def test_detailed_metrics_include_mel_recall_and_specificity():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    probs = np.eye(2)[preds]
    metrics = detailed_metrics(labels, preds, ["MEL", "BCC"], probs)

    assert metrics["mel_recall"] == pytest.approx(0.5)
    assert metrics["mel_specificity"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(11 / 15)


def test_probability_metrics_reject_invalid_rows():
    with pytest.raises(ValueError, match="sum to one"):
        probability_metrics(np.array([0]), np.array([[0.2, 0.2]]), 2)


def test_ece_handles_confidence_one():
    assert expected_calibration_error(np.array([0]), np.array([[1.0, 0.0]])) == 0.0

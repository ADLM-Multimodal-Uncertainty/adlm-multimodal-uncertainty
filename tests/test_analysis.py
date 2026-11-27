"""dEU analysis on synthetic ensemble outputs (no training, no images)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from src import CLASSES
from src.analysis import (
    apply_temperature,
    classification_metrics,
    compare,
    expected_calibration_error,
    fit_temperature,
    load_arm,
)
from src.uq.deep_ensemble import uncertainty

from .conftest import REPO

N_PER_FOLD, N_MEMBERS = 60, 3


def fake_ensemble_run(root: Path, name: str, fold: int, member_probs: np.ndarray,
                      img_ids, patient_ids, labels, tabular: bool,
                      val: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """The files `src.train` writes for a finished deep ensemble run."""
    run_dir = root / f"{name}-patient_kfold-f{fold}-s42"
    run_dir.mkdir(parents=True)
    cfg = {
        "name": name, "fold": fold, "seed": 42,
        "split": {"_target_": "src.data.splits.PatientKFoldSplit", "n_folds": 2},
        "ensemble": {"seeds": list(range(N_MEMBERS))}, "train": {"weighted_loss": False},
        "model": {"image_encoder": {"freeze": False},
                  "tabular_encoder": {"hidden_dims": [64]} if tabular else None},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    probs = member_probs.mean(axis=0)
    total, aleatoric, epistemic = uncertainty(member_probs)
    predictions = pd.DataFrame({"patient_id": patient_ids, "img_id": img_ids,
                                "diagnostic": [CLASSES[i] for i in labels]})
    predictions["total_uncertainty"] = total
    predictions["aleatoric_uncertainty"] = aleatoric
    predictions["epistemic_uncertainty"] = epistemic
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(classification_metrics(labels, probs.argmax(axis=1)))
    )
    np.savez_compressed(
        run_dir / "test_member_predictions.npz", probabilities=member_probs, labels=labels,
        img_ids=np.asarray(img_ids, dtype=str), patient_ids=np.asarray(patient_ids, dtype=str),
        classes=np.asarray(CLASSES), seeds=np.arange(N_MEMBERS),
    )
    if val is not None:
        val_probs, val_labels = val
        n = len(val_labels)
        np.savez_compressed(
            run_dir / "val_member_predictions.npz", probabilities=val_probs, labels=val_labels,
            img_ids=np.asarray([f"V{fold}_{i}" for i in range(n)]),
            patient_ids=np.asarray([f"VPAT_{fold}{i}" for i in range(n)]),
            classes=np.asarray(CLASSES), seeds=np.arange(N_MEMBERS),
        )


def random_probs(rng, n, spread):
    logits = rng.normal(size=(N_MEMBERS, n, len(CLASSES))) * spread
    logits += rng.normal(size=(1, n, len(CLASSES))) * 2
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.fixture
def runs_root(tmp_path):
    """Image-only members disagree more than multimodal ones, so dEU > 0."""
    rng = np.random.default_rng(0)
    root = tmp_path / "runs"
    for fold in (0, 1):
        img_ids = [f"PAT_{fold}{i // 2:03d}_{i}.png" for i in range(N_PER_FOLD)]
        patients = [f"PAT_{fold}{i // 2:03d}" for i in range(N_PER_FOLD)]
        labels = np.arange(N_PER_FOLD) % len(CLASSES)
        val_labels = np.arange(30) % len(CLASSES)
        fake_ensemble_run(root, "img", fold, random_probs(rng, N_PER_FOLD, 2.0),
                          img_ids, patients, labels, tabular=False,
                          val=(random_probs(rng, 30, 2.0), val_labels))
        order = rng.permutation(N_PER_FOLD)  # the other arm saves images in another order
        fake_ensemble_run(root, "mm", fold, random_probs(rng, N_PER_FOLD, 0.3)[:, order],
                          [img_ids[i] for i in order], [patients[i] for i in order],
                          labels[order], tabular=True,
                          val=(random_probs(rng, 30, 0.3), val_labels))
    return root


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, len(CLASSES), 200)
    preds = np.where(rng.random(200) < 0.6, labels, rng.integers(0, len(CLASSES), 200))
    ours = classification_metrics(labels, preds)
    assert np.isclose(ours["balanced_accuracy"], balanced_accuracy_score(labels, preds))
    assert np.isclose(ours["macro_f1"], f1_score(labels, preds, labels=range(6),
                                                 average="macro", zero_division=0))
    np.testing.assert_allclose(list(ours["recall"].values()),
                               recall_score(labels, preds, labels=range(6), average=None))


def test_ece_limits():
    labels = np.array([0, 1, 2, 3])
    assert expected_calibration_error(np.eye(6)[labels], labels) == 0
    assert np.isclose(expected_calibration_error(np.eye(6)[[1, 2, 3, 4]], labels), 1)


def test_compare_pools_folds_and_aligns(runs_root):
    image_only, multimodal = load_arm(runs_root, "img"), load_arm(runs_root, "mm")
    summary, per_image = compare(image_only, multimodal, n_boot=200, seed=0)
    assert len(per_image) == 2 * N_PER_FOLD and per_image.img_id.is_unique
    assert summary["warnings"] == []
    raw = summary["uncalibrated"]

    # Every number is recomputable from the saved per-image table.
    np.testing.assert_allclose(
        per_image.delta_epistemic, per_image.epistemic_image_only - per_image.epistemic_multimodal
    )
    d = raw["delta_epistemic"]
    assert np.isclose(d["mean"], per_image.delta_epistemic.mean())
    assert d["mean"] > 0 and d["ci95"][0] <= d["mean"] <= d["ci95"][1]
    assert d["n_patients"] == per_image.patient_id.nunique()
    assert set(d["per_class"]) == set(CLASSES)
    pooled = raw["metrics"]["multimodal"]["pooled"]
    labels = per_image.diagnostic.map(CLASSES.index).to_numpy()
    preds = per_image.pred_multimodal.map(CLASSES.index).to_numpy()
    assert np.isclose(pooled["balanced_accuracy"], balanced_accuracy_score(labels, preds))
    assert set(raw["metrics"]["image_only"]["per_fold"]) == {0, 1}
    assert set(pooled["mean_epistemic_per_class"]) == set(CLASSES)

    # Temperature scaling ran: one temperature per member and fold, recomputable columns.
    assert summary["temperature_scaled"] is not None
    assert all(len(t) == N_MEMBERS for t in summary["temperatures"]["image_only"].values())
    np.testing.assert_allclose(
        per_image.delta_epistemic_ts,
        per_image.epistemic_image_only_ts - per_image.epistemic_multimodal_ts,
    )


def test_temperature_scaling_is_skipped_without_validation_predictions(runs_root):
    for path in runs_root.glob("*/val_member_predictions.npz"):
        path.unlink()
    summary, _ = compare(load_arm(runs_root, "img"), load_arm(runs_root, "mm"), 50, 0)
    assert summary["temperature_scaled"] is None
    assert any("temperature scaling skipped" in w for w in summary["warnings"])


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4000, len(CLASSES))) * 1.5
    true_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(len(CLASSES), p=p) for p in true_probs])
    overconfident = apply_temperature(true_probs, 1 / 3)  # logits multiplied by 3
    assert fit_temperature(overconfident, labels) == pytest.approx(3, rel=0.15)
    np.testing.assert_allclose(apply_temperature(true_probs, 1.0), true_probs, atol=1e-12)


def test_mismatched_images_are_rejected(runs_root, tmp_path):
    rng = np.random.default_rng(2)
    labels = np.zeros(5, dtype=int)
    for fold in (0, 1):
        fake_ensemble_run(runs_root, "other", fold, random_probs(rng, 5, 1.0),
                          [f"X{fold}_{i}" for i in range(5)], [f"X{fold}"] * 5, labels, True)
    with pytest.raises(ValueError, match="same images"):
        compare(load_arm(runs_root, "img"), load_arm(runs_root, "other"), n_boot=10, seed=0)


def test_cli_writes_report(runs_root, tmp_path):
    out = tmp_path / "analysis"
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis", "img:mm", "--runs-root", str(runs_root),
         "--out", str(out), "--n-boot", "50"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ("per_image.csv", "summary.json", "summary.md"):
        assert (out / "img_vs_mm" / name).is_file()
    report = (out / "report.md").read_text()
    assert "Mean ΔEU" in report and "## Overview" in report
    assert "After temperature scaling" in report

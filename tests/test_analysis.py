"""Hand checks for the scoring rules and the standardized report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src import CLASSES, meaningful_overrides
from src.analysis.effects import quadrant_counts, reduction_stats
from src.analysis.pipeline import run_analysis
from src.data.transforms import AugmentationConfig, image_transform
from src.metrics import PROBABILITY_FLOOR, brier_score, nll
from src.train import class_weights
from src.uq.deep_ensemble import uncertainty

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_meaningful_overrides_keep_only_model_changes():
    kept = meaningful_overrides(
        [
            "fold=1",
            "data.root=/tmp/pad",
            "train.optimizer.lr=0.0001",
            "data.augmentation=crop_zoom",
        ]
    )
    assert kept == ["train.optimizer.lr=0.0001", "data.augmentation=crop_zoom"]


def test_nll_and_brier_match_a_hand_calculation():
    labels = np.array([0, 1])
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert nll(labels, probs) == pytest.approx(0.5 * np.log(2.0))
    # Second image: (0.5 - 0)^2 + (0.5 - 1)^2 = 0.5; first image is exact.
    assert brier_score(labels, probs) == pytest.approx(0.25)


def test_perfect_prediction_has_zero_nll_without_an_upper_clip():
    labels = np.array([0])
    assert nll(labels, np.array([[1.0, 0.0]])) == 0.0


def test_class_weights_keep_inverse_frequency_ratios_and_mean_one():
    labels = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
    assert class_weights(labels, "cpu", 0.0) is None
    weights = class_weights(labels, "cpu", 1.0).numpy()
    raw = len(labels) / (len(CLASSES) * np.bincount(labels, minlength=len(CLASSES)))
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] / weights[1] == pytest.approx(raw[0] / raw[1])
    with pytest.raises(ValueError, match="SEK"):
        class_weights(np.array([0, 1, 2, 3, 4]), "cpu", 1.0)


def test_baseline_augmentation_is_geometric_and_evaluation_is_deterministic():
    baseline = AugmentationConfig(horizontal_flip=0.5, vertical_flip=0.5, rotation_degrees=15)
    training = [type(op).__name__ for op in image_transform(32, True, baseline).transforms]
    evaluation = [type(op).__name__ for op in image_transform(32, False, baseline).transforms]
    assert "RandomHorizontalFlip" in training
    assert "RandomVerticalFlip" in training
    assert "RandomRotation" in training
    assert "ColorJitter" not in training
    assert "RandomResizedCrop" not in training
    assert not any(name.startswith("Random") for name in evaluation)


def test_identical_uncertainty_has_no_disagreement_and_zero_reference_is_unavailable():
    total, aleatoric, epistemic = uncertainty(np.full((3, 4, 6), 1.0 / 6.0))
    np.testing.assert_allclose(epistemic, 0, atol=1e-12)
    np.testing.assert_allclose(total, aleatoric)
    stats = reduction_stats(np.array([0.2, 0.4]), np.array([0.2, 0.4]))
    assert stats["mean_nats"] == 0.0
    assert stats["share_reduced_pct"] == 0.0
    assert reduction_stats(np.zeros(3), np.ones(3))["relative_pct"] == "unavailable"


def test_correctness_groups_cover_every_image():
    counts = dict(
        quadrant_counts(
            np.array([True, False, True, False]),
            np.array([True, True, False, False]),
        )
    )
    assert counts == {
        "both_correct": 1,
        "reference_wrong_candidate_correct": 1,
        "reference_correct_candidate_wrong": 1,
        "both_wrong": 1,
    }
    assert sum(counts.values()) == 4


def _one_hot(prediction: int) -> np.ndarray:
    row = np.zeros(len(CLASSES), dtype=np.float64)
    row[prediction] = 1.0
    return row


def _write_split(fold_dir, split: str, rows: list[tuple[str, str, int, int, int]]) -> None:
    labels = np.array([row[2] for row in rows])
    img_ids = np.array([row[0] for row in rows])
    patient_ids = np.array([row[1] for row in rows])
    # Two agreeing members, so epistemic uncertainty is exactly zero.
    members = [
        np.stack([_one_hot(row[3]) for row in rows]),
        np.stack([_one_hot(row[3]) for row in rows]),
    ]
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fold_dir / f"{split}_member_predictions.npz",
        probabilities=np.stack(members),
        labels=labels,
        img_ids=img_ids,
        patient_ids=patient_ids,
        classes=np.asarray(CLASSES),
        seeds=np.array([0, 1]),
    )


def _write_history(fold_dir) -> None:
    epochs = {0: [1, 2], 1: [1, 2, 3]}
    for seed, values in epochs.items():
        member = fold_dir / "members" / f"seed_{seed}"
        member.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": values,
                "val_loss": [1.0 / epoch for epoch in values],
                "val_accuracy": [0.5] * len(values),
            }
        ).to_csv(member / "history.csv", index=False)


def _write_condition(runs: Path, group: str, prediction_column: int) -> None:
    """Rows are img, patient, label, reference prediction, candidate prediction."""
    folds = {
        0: [("a", "p0", 0, 0, 0), ("b", "p1", 0, 1, 0)],
        1: [("c", "p2", 0, 0, 1), ("d", "p3", 0, 1, 1)],
    }
    for fold, rows in folds.items():
        fold_dir = runs / group / f"fold_{fold}"
        chosen = [
            (item[0], item[1], item[2], item[prediction_column], item[prediction_column])
            for item in rows
        ]
        _write_split(fold_dir, "test", chosen)
        _write_split(fold_dir, "val", chosen[:1])
        _write_history(fold_dir)


def test_single_setup_report_matches_the_hand_calculated_example(tmp_path):
    runs = tmp_path / "runs"
    _write_condition(runs, "image_only", 3)
    _write_condition(runs, "multimodal", 4)
    cfg = OmegaConf.create(
        {
            "runs_root": str(runs),
            "analysis_root": str(tmp_path / "result-analysis"),
            "experiment_id": "hand_example",
            "eval_split": "test",
            "n_folds": 2,
            "bootstrap_samples": 20,
            "bootstrap_seed": 0,
            "reference": "image_only",
            "candidate": "multimodal",
            "reference_label": None,
            "candidate_label": None,
            "baseline_label": "Baseline",
            "baseline_image": None,
            "baseline_multimodal": None,
            "candidate_image": None,
            "candidate_multimodal": None,
        }
    )
    output = run_analysis(cfg)
    paired = pd.read_csv(output / "data" / "paired_images.csv")
    assert len(paired) == 4
    counts = dict(
        quadrant_counts(
            paired["reference_correct"].to_numpy(),
            paired["candidate_correct"].to_numpy(),
        )
    )
    assert sum(counts.values()) == 4
    assert counts["both_correct"] == 1
    effects = pd.read_csv(output / "tables" / "paired_effects.csv")
    assert effects.loc[0, "delta_balanced_accuracy_pp"] == pytest.approx(0.0)
    assert effects.loc[0, "mean_epistemic_uncertainty_reduction_nats"] == pytest.approx(0.0)
    assert effects.loc[0, "relative_epistemic_uncertainty_reduction_pct"] == "unavailable"
    performance = pd.read_csv(output / "tables" / "performance_and_uncertainty.csv")
    pooled = performance[performance["fold"] == "pooled"]
    wrong_nll = 0.5 * -np.log(PROBABILITY_FLOOR)
    assert pooled["nll_nats"].tolist() == pytest.approx([wrong_nll, wrong_nll])
    assert pooled["brier"].tolist() == pytest.approx([1.0, 1.0])
    history = pd.read_csv(output / "data" / "validation_history.csv")
    assert history["epoch"].max() == 3
    assert 4 not in set(history["epoch"])
    assert len(history[history["epoch"] == 3]) == 4
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "Share reduced (%)" in report
    assert "mean_epistemic_uncertainty_reduction_nats" not in report
    assert "Δ bal. acc. (pp)" in report
    for name in (
        "paired_eu_scatter_primary.png",
        "paired_deu_distribution_primary.png",
        "paired_deu_candles_primary.png",
        "confusion_matrices.png",
        "paired_effect_estimates.png",
        "validation_curves.png",
    ):
        blob = (output / "figures" / name).read_bytes()
        assert blob.startswith(PNG_MAGIC)
    manifest = OmegaConf.load(output / "manifest.yaml")
    assert manifest.eval_split == "test"
    assert manifest.metric_conventions.nll_floor == 1e-12
    summary = pd.read_csv(tmp_path / "result-analysis" / "summary" / "experiment_uncertainty.csv")
    assert set(summary["experiment_id"]) == {"hand_example"}
    summary_figure = tmp_path / "result-analysis" / "summary" / "total_vs_aleatoric_uncertainty.png"
    assert summary_figure.read_bytes().startswith(PNG_MAGIC)

    validation = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    validation.experiment_id = "hand_example_validation"
    validation.eval_split = "val"
    validation_output = run_analysis(validation)
    validation_paired = pd.read_csv(validation_output / "data" / "paired_images.csv")
    assert len(validation_paired) == 2
    summary = pd.read_csv(tmp_path / "result-analysis" / "summary" / "experiment_uncertainty.csv")
    assert set(summary["experiment_id"]) == {"hand_example"}


def test_baseline_versus_candidate_report_writes_four_comparisons(tmp_path):
    runs = tmp_path / "runs"
    for group in ("base_image", "base_meta", "cand_image", "cand_meta"):
        _write_condition(runs, group, 3)
    output = run_analysis(
        OmegaConf.create(
            {
                "runs_root": str(runs),
                "analysis_root": str(tmp_path / "result-analysis"),
                "experiment_id": "crop_versus_baseline",
                "eval_split": "val",
                "n_folds": 2,
                "bootstrap_samples": 5,
                "bootstrap_seed": 1,
                "reference": None,
                "candidate": None,
                "reference_label": None,
                "candidate_label": "Crop and zoom",
                "baseline_label": "Baseline",
                "baseline_image": "base_image",
                "baseline_multimodal": "base_meta",
                "candidate_image": "cand_image",
                "candidate_multimodal": "cand_meta",
            }
        )
    )
    effects = pd.read_csv(output / "tables" / "paired_effects.csv")
    assert list(effects["comparison"]) == [
        "Baseline modality effect",
        "Crop and zoom modality effect",
        "Image-only: Baseline -> Crop and zoom",
        "Image + Metadata: Baseline -> Crop and zoom",
    ]
    assert effects["delta_macro_f1_pp"].abs().max() == 0
    assert len(list((output / "figures").glob("paired_eu_scatter_*.png"))) == 4
    assert len(list((output / "figures").glob("paired_deu_candles_*.png"))) == 4
    assert "Crop and zoom modality effect" in (output / "REPORT.md").read_text(encoding="utf-8")
    assert not (tmp_path / "result-analysis" / "summary" / "experiment_uncertainty.csv").exists()

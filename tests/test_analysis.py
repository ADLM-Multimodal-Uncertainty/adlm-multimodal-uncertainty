import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.analysis import CLINICAL_BLOCK_COLUMNS, PairedAnalysis, bootstrap_mean_ci


def _write_history(member_dir: Path, loss_offset: float = 0.0) -> None:
    member_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 1.2 + loss_offset,
                "val_loss": 1.1 + loss_offset,
                "train_accuracy": 0.55,
                "train_balanced_accuracy": 0.52,
                "train_macro_f1": 0.50,
                "val_accuracy": 0.50,
                "val_balanced_accuracy": 0.48,
                "val_macro_f1": 0.46,
            },
            {
                "epoch": 2,
                "train_loss": 0.8 + loss_offset,
                "val_loss": 0.9 + loss_offset,
                "train_accuracy": 0.78,
                "train_balanced_accuracy": 0.75,
                "train_macro_f1": 0.74,
                "val_accuracy": 0.72,
                "val_balanced_accuracy": 0.70,
                "val_macro_f1": 0.68,
            },
        ]
    ).to_csv(member_dir / "history.csv", index=False)


def _write_artifact(
    path: Path,
    probabilities: np.ndarray,
    tabular_encoder: dict | None,
    *,
    name: str = "r18_full",
    fold: int = 0,
    arch: str = "resnet18",
    train_fraction: float = 1.0,
    seeds: tuple[int, ...] = (0, 1, 2),
    labels: list[int] | None = None,
    img_ids: list[str] | None = None,
    patient_ids: list[str] | None = None,
    metadata_csv: Path | None = None,
    write_histories: bool = True,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    labels_arr = np.array([0, 1, 0, 1]) if labels is None else np.asarray(labels)
    img_arr = (
        np.array([f"I{fold}_{i}" for i in range(len(labels_arr))])
        if img_ids is None
        else np.asarray(img_ids)
    )
    pat_arr = (
        np.array([f"P{fold}_{i}" for i in range(len(labels_arr))])
        if patient_ids is None
        else np.asarray(patient_ids)
    )
    config = OmegaConf.create(
        {
            "name": name,
            "fold": fold,
            "split": {"n_folds": 5},
            "data": {
                "train_fraction": train_fraction,
                "train_subset_seed": 42,
                "metadata_csv": str(metadata_csv) if metadata_csv else None,
            },
            "train": {"epochs": 2},
            "model": {
                "image_encoder": {"arch": arch, "freeze": False},
                "tabular_encoder": tabular_encoder,
            },
            "ensemble": {"seeds": list(seeds)},
        }
    )
    OmegaConf.save(config, path / "config.yaml")
    np.savez_compressed(
        path / "test_member_predictions.npz",
        probabilities=probabilities,
        labels=labels_arr,
        img_ids=img_arr,
        patient_ids=pat_arr,
        classes=np.array(["A", "MEL"]),
        seeds=np.array(seeds),
    )
    if write_histories:
        for idx, seed in enumerate(seeds):
            _write_history(path / "members" / f"seed_{seed}", loss_offset=0.05 * idx)


def _make_probs(spread: float) -> np.ndarray:
    """Three ensemble members whose disagreement scales monotonically with `spread`."""
    base = np.array([[0.85, 0.15], [0.15, 0.85], [0.80, 0.20], [0.20, 0.80]])
    shifted_up = np.clip(base + np.array([-spread, spread]), 0.02, 0.98)
    shifted_up /= shifted_up.sum(axis=1, keepdims=True)
    shifted_dn = np.clip(base + np.array([spread, -spread]), 0.02, 0.98)
    shifted_dn /= shifted_dn.sum(axis=1, keepdims=True)
    return np.stack([base, shifted_up, shifted_dn])


def _write_metadata(path: Path, folds: tuple[int, ...] = (0, 1)) -> None:
    rows = []
    for fold in folds:
        for i in range(4):
            missing_block = i >= 2
            row: dict[str, object] = {
                "img_id": f"I{fold}_{i}",
                "patient_id": f"P{fold}_{i}",
                "lesion_id": i // 2,
                "diagnostic": "A" if i % 2 == 0 else "MEL",
            }
            for col in CLINICAL_BLOCK_COLUMNS:
                row[col] = np.nan if missing_block else "True"
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _analysis(
    tmp_path: Path,
    multimodal_seeds: tuple[int, ...] = (0, 1, 2),
    multimodal_labels: list[int] | None = None,
) -> PairedAnalysis:
    meta_path = tmp_path / "metadata.csv"
    _write_metadata(meta_path, folds=(0,))
    image_probs = _make_probs(0.35)
    multimodal_probs = _make_probs(0.05)
    image_dir = tmp_path / "image"
    multimodal_dir = tmp_path / "multimodal"
    _write_artifact(image_dir, image_probs, None, metadata_csv=meta_path)
    _write_artifact(
        multimodal_dir,
        multimodal_probs,
        {"hidden_dims": [64]},
        seeds=multimodal_seeds,
        labels=multimodal_labels,
        metadata_csv=meta_path,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "conditions": {
                    "r18_full": {
                        "image_run_dirs": [str(image_dir)],
                        "multimodal_run_dirs": [str(multimodal_dir)],
                    }
                }
            }
        )
    )
    return PairedAnalysis(
        manifest=str(manifest),
        output_dir=str(tmp_path / "analysis"),
        bootstrap_samples=100,
        seed=8,
    )


def test_paired_analysis_writes_all_per_condition_artifacts(tmp_path):
    output = _analysis(tmp_path).run()

    comparison = pd.read_csv(output / "condition_comparison.csv")
    assert comparison.loc[0, "mean_deu"] > 0
    assert comparison.loc[0, "architecture"] == "resnet18"
    assert comparison.loc[0, "train_fraction"] == 1.0
    patient = pd.read_csv(output / "r18_full" / "per_patient.csv")
    assert len(patient) == 4
    assert (patient["depistemic"] > 0).all()

    expected_condition_files = (
        "per_image.csv",
        "per_patient.csv",
        "uncertainty_summary.csv",
        "per_fold_summary.csv",
        "training_curves.csv",
        "error_detection_summary.csv",
        "risk_coverage.csv",
        "aurc_summary.csv",
        "ensemble_size_stability.csv",
        "missing_metadata_summary.csv",
        "classification_metrics.json",
        "uncertainty.png",
        "paired_eu_scatter.png",
        "correct_vs_incorrect_eu.png",
        "risk_coverage.png",
        "classification.png",
        "calibration.png",
        "fold_comparison.png",
        "ensemble_size_stability.png",
        "loss_accuracy_curves.png",
        "missing_metadata_stratification.png",
    )
    for filename in expected_condition_files:
        file_path = output / "r18_full" / filename
        assert file_path.exists() and file_path.stat().st_size > 0, filename

    missing_df = pd.read_csv(output / "r18_full" / "missing_metadata_summary.csv")
    assert {"clinical_block_present", "clinical_block_missing", "all (lesion-level)"}.issubset(
        set(missing_df["stratum"])
    )
    stability = pd.read_csv(output / "r18_full" / "ensemble_size_stability.csv")
    assert stability["n_members"].tolist() == [2, 3]


def test_auto_discovery_multi_fold_positive_control_and_markdown_tables(tmp_path):
    runs_root = tmp_path / "runs"
    meta_path = tmp_path / "metadata.csv"
    _write_metadata(meta_path, folds=(0, 1))

    setups = [
        ("eu_baseline_r18", "resnet18", 1.0, 0.15, 0.05),
        ("eu_size_r18_f050", "resnet18", 0.5, 0.25, 0.12),
        ("eu_size_r18_f025", "resnet18", 0.25, 0.40, 0.22),
        ("eu_resnet50", "resnet50", 1.0, 0.14, 0.04),
    ]
    for prefix, arch, frac, img_spread, mm_spread in setups:
        for fold in (0, 1):
            _write_artifact(
                runs_root / f"{prefix}_image_only-patient_kfold-f{fold}-s42",
                _make_probs(img_spread),
                None,
                name=f"{prefix}_image_only",
                fold=fold,
                arch=arch,
                train_fraction=frac,
                metadata_csv=meta_path,
            )
            _write_artifact(
                runs_root / f"{prefix}_multimodal-patient_kfold-f{fold}-s42",
                _make_probs(mm_spread),
                {"hidden_dims": [64]},
                name=f"{prefix}_multimodal",
                fold=fold,
                arch=arch,
                train_fraction=frac,
                metadata_csv=meta_path,
            )

    output = PairedAnalysis(
        runs_root=str(runs_root),
        output_dir=str(tmp_path / "analysis"),
        bootstrap_samples=200,
        seed=42,
    ).run()

    for filename in (
        "resolved_manifest.json",
        "condition_comparison.csv",
        "condition_comparison.json",
        "condition_comparison.png",
        "all_folds_comparison.csv",
        "cross_setup_fold_eu.png",
        "eu_measures_comparison.png",
        "eu_measures_agreement.csv",
        "all_training_curves.csv",
        "cross_setup_learning_curves.png",
        "train_fraction.png",
        "positive_control_summary.csv",
        "summary_tables.md",
    ):
        file_path = output / filename
        assert file_path.exists() and file_path.stat().st_size > 0, filename

    pos = pd.read_csv(output / "positive_control_summary.csv")
    assert set(pos["arm"]) == {"image", "multimodal"}
    assert (pos["monotonic_decrease"]).all()
    assert (pos["status"] == "PASS").all()

    md_text = (output / "summary_tables.md").read_text()
    assert "## 1. Overall Setup Comparison" in md_text
    assert "## 2. Per-Fold EU and Classification" in md_text
    assert "Experiment 6 (ResNet18 vs ResNet50" in md_text
    assert "Experiment 4 Positive Control" in md_text


@pytest.mark.parametrize(
    ("seeds", "labels", "message"),
    [
        ((0, 1, 9), None, "seeds"),
        ((0, 1, 2), [0, 0, 0, 1], "labels"),
    ],
)
def test_paired_analysis_rejects_incompatible_arms(tmp_path, seeds, labels, message):
    with pytest.raises(ValueError, match=message):
        _analysis(tmp_path, seeds, labels).run()


def test_patient_bootstrap_is_deterministic():
    values = np.array([0.1, 0.2, 0.3])
    assert bootstrap_mean_ci(values, n_samples=100, seed=4) == bootstrap_mean_ci(
        values, n_samples=100, seed=4
    )

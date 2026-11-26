"""`python -m src.train experiment=...` end to end on synthetic data, as scripts/local/run.sh."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from src import CLASSES
from src.data.dataset import build_fold
from src.data.splits import PatientHoldoutSplit
from src.data.transforms import image_transform

from .conftest import REPO


@pytest.mark.parametrize("experiment", ["draft_baseline", "draft_image_only"])
def test_experiment_trains_and_evaluates(experiment, synthetic_root, tmp_path):
    epochs = 2
    result = subprocess.run(
        [
            sys.executable, "-m", "src.train",
            f"experiment={experiment}",
            f"data.root={synthetic_root}",
            f"output_root={tmp_path / 'outputs'}",
            "data.image_size=32",
            f"train.epochs={epochs}",
            "train.batch_size=16",
            "model.image_encoder.pretrained=false",
            "model.image_encoder.freeze=false",
            "device=cpu",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "outputs" / "runs" / f"{experiment}-patient_holdout-f0-s42"
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert (cfg.model.get("tabular_encoder") is None) == (experiment == "draft_image_only")
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "curves.png").stat().st_size > 0
    assert (run_dir / "confusion_matrix.png").stat().st_size > 0
    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == epochs
    for column in ("train_loss", "val_loss", "train_macro_f1", "val_macro_f1"):
        assert history[column].notna().all(), column

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    assert len(predictions) == metrics["n_images"] > 0
    probs = predictions[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert abs(probs.sum(axis=1) - 1).max() < 1e-4
    assert 0.0 <= metrics["macro_f1"] <= 1.0


@pytest.mark.parametrize("mode", ["multimodal", "image_only"])
@pytest.mark.parametrize("n_folds", [2, 5])
def test_five_fold_multirun(mode, n_folds, synthetic_root, tmp_path):
    df = pd.read_csv(synthetic_root / "metadata.csv")
    # Patient suffix indexes are balanced within each synthetic diagnosis.
    assignment = df[["img_id", "patient_id"]].copy()
    assignment["fold"] = assignment["patient_id"].str[-2:].astype(int) % 5
    csv_path = tmp_path / "folds.csv"
    assignment.to_csv(csv_path, index=False)
    experiment = f"patient_5fold_{mode}"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.train", "-m",
            f"experiment={experiment}",
            f"split.n_folds={n_folds}",
            *([f"split.csv={csv_path}"] if n_folds == 5 else []),
            f"data.root={synthetic_root}", f"output_root={tmp_path / 'outputs'}",
            "data.image_size=32", "data.subset=6", "train.epochs=1",
            "train.batch_size=6", "model.image_encoder.pretrained=false",
            "model.image_encoder.freeze=false", "device=cpu",
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tested = []
    for fold in range(n_folds):
        run_dir = tmp_path / "outputs" / "runs" / f"{experiment}-patient_kfold-f{fold}-s42"
        cfg = OmegaConf.load(run_dir / "config.yaml")
        assert cfg.fold == fold
        assert cfg.split.n_folds == n_folds
        assert (cfg.model.get("tabular_encoder") is None) == (mode == "image_only")
        assert (run_dir / "best.pt").exists()
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        tested.append(set(predictions["patient_id"]))
    for i, patients in enumerate(tested):
        assert all(not patients.intersection(other) for other in tested[i + 1:])


def test_image_transform_presets_and_validation():
    img = Image.fromarray(np.full((48, 48, 3), 120, dtype=np.uint8))
    for preset in ("none", "flip", "derm"):
        train_out = image_transform(32, training=True, augmentation=preset)(img)
        val_out = image_transform(32, training=False, augmentation=preset)(img)
        assert train_out.shape == (3, 32, 32)
        assert val_out.shape == (3, 32, 32)
    with pytest.raises(ValueError, match="unknown augmentation preset"):
        image_transform(32, training=True, augmentation="invalid")


def test_permute_metadata_shuffles_deterministically_per_split(synthetic_root):
    data_yaml = OmegaConf.load(REPO / "configs" / "data" / "pad_ufes20.yaml")
    root_cfg = OmegaConf.create({"data": data_yaml})
    root_cfg.data.root = str(synthetic_root)
    root_cfg.data.image_size = 32
    split = PatientHoldoutSplit(val_size=0.2, test_size=0.2, seed=42)

    clean = build_fold(root_cfg.data, split, fold=0)
    perm_root = OmegaConf.create(OmegaConf.to_container(root_cfg, resolve=True))
    perm_root.data.permute_metadata = True
    perm_root.data.permute_metadata_seed = 7
    perm_a = build_fold(perm_root.data, split, fold=0)
    perm_b = build_fold(perm_root.data, split, fold=0)

    for name in ("train", "val", "test"):
        clean_feats = getattr(clean, name).features
        perm_a_feats = getattr(perm_a, name).features
        perm_b_feats = getattr(perm_b, name).features
        assert np.array_equal(perm_a_feats, perm_b_feats)
        assert not np.array_equal(clean_feats, perm_a_feats)
        # Same multiset of rows (marginal distribution preserved within split)
        assert np.allclose(np.sort(clean_feats, axis=0), np.sort(perm_a_feats, axis=0))


def test_eu_v2_training_supports_val_loss_scheduler_early_stopping_and_derm_aug(
    synthetic_root, tmp_path
):
    result = subprocess.run(
        [
            sys.executable, "-m", "src.train",
            "experiment=eu_v2_r18_multimodal",
            "~ensemble",
            "split.n_folds=2",
            f"data.root={synthetic_root}",
            f"output_root={tmp_path / 'outputs'}",
            "data.image_size=32",
            "train.epochs=4",
            "train.early_stopping_patience=2",
            "train.batch_size=16",
            "model.image_encoder.pretrained=false",
            "device=cpu",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "outputs" / "runs" / "eu_v2_r18_multimodal-patient_kfold-f0-s42"
    history = pd.read_csv(run_dir / "history.csv")
    assert "lr" in history.columns
    assert 1 <= len(history) <= 4
    assert (run_dir / "best.pt").exists()
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

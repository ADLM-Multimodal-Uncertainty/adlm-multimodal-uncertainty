"""`python -m src.train experiment=...` end to end on synthetic data, as scripts/local/run.sh."""

import json
import subprocess
import sys

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src import CLASSES

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
def test_five_fold_multirun(mode, synthetic_root, tmp_path):
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
            f"experiment={experiment}", f"split.csv={csv_path}",
            f"data.root={synthetic_root}", f"output_root={tmp_path / 'outputs'}",
            "data.image_size=32", "data.subset=6", "train.epochs=1",
            "train.batch_size=6", "model.image_encoder.pretrained=false",
            "model.image_encoder.freeze=false", "device=cpu",
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tested = []
    for fold in range(5):
        run_dir = tmp_path / "outputs" / "runs" / f"{experiment}-patient_kfold-f{fold}-s42"
        cfg = OmegaConf.load(run_dir / "config.yaml")
        assert cfg.fold == fold
        assert cfg.split.n_folds == 5
        assert (cfg.model.get("tabular_encoder") is None) == (mode == "image_only")
        assert (run_dir / "best.pt").exists()
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        tested.append(set(predictions["patient_id"]))
    for i, patients in enumerate(tested):
        assert all(not patients.intersection(other) for other in tested[i + 1:])

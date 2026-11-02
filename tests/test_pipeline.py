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
    assert len(pd.read_csv(run_dir / "history.csv")) == epochs

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    assert len(predictions) == metrics["n_images"] > 0
    probs = predictions[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert abs(probs.sum(axis=1) - 1).max() < 1e-4
    assert 0.0 <= metrics["macro_f1"] <= 1.0

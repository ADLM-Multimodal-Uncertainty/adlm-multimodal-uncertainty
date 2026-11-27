"""Entropy limits and saved ensemble pipeline round trips."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluate import evaluate_run
from src.train import improved
from src.uq.deep_ensemble import DeepEnsemble, uncertainty

from .conftest import REPO


@pytest.mark.parametrize("seeds", [[0], [0, 0], [-1, 2], [0, 1.5]])
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        DeepEnsemble(seeds)


def test_first_finite_validation_loss_replaces_the_sentinel():
    assert improved(1.5, float("inf"), "min", 0.0)
    assert improved(0.4, 1.5, "min", 0.0)
    assert not improved(1.5, 1.0, "min", 0.0)
    assert not improved(float("nan"), 1.0, "min", 0.0)


def test_entropy_decomposition_is_finite_for_float32_zeros():
    probs = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    total, aleatoric, epistemic = uncertainty(probs)
    assert np.isfinite(total).all()
    assert np.isfinite(aleatoric).all()
    assert np.isfinite(epistemic).all()
    np.testing.assert_allclose(total, np.log(2))
    np.testing.assert_allclose(aleatoric, 0, atol=1e-6)
    np.testing.assert_allclose(epistemic, total)


def test_entropy_decomposition():
    total, aleatoric, epistemic = uncertainty(np.array([[[1., 0.]], [[0., 1.]]]))
    np.testing.assert_allclose(total, np.log(2))
    np.testing.assert_allclose(aleatoric, 0)
    np.testing.assert_allclose(epistemic, total)
    total, aleatoric, epistemic = uncertainty(np.full((2, 3, 2), .5))
    np.testing.assert_allclose(total, aleatoric)
    np.testing.assert_allclose(epistemic, 0, atol=1e-12)


@pytest.mark.parametrize("experiment", ["baseline_multimodal", "baseline_image_only"])
def test_ensemble_pipeline(experiment, synthetic_root, tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "src.train", f"experiment={experiment}",
        "ensemble.seeds=[3,7]", f"data.root={synthetic_root}",
        f"output_root={tmp_path / 'outputs'}", "data.image_size=32",
        "train.epochs=1", "train.batch_size=16", "device=cpu",
        "model.image_encoder.pretrained=false", "model.image_encoder.freeze=false",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "outputs" / "runs" / experiment / "fold_0"
    predictions = pd.read_csv(run_dir / 'test_predictions.csv')
    with np.load(run_dir / 'test_member_predictions.npz', allow_pickle=False) as saved:
        assert saved['probabilities'].shape == (2, len(predictions), 6)
        np.testing.assert_array_equal(saved['seeds'], [3, 7])
        np.testing.assert_allclose(saved['probabilities'].sum(axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(
            saved['probabilities'].mean(axis=0),
            predictions[[f'p_{c}' for c in saved['classes']]],
        )
    np.testing.assert_allclose(
        predictions.total_uncertainty,
        predictions.aleatoric_uncertainty + predictions.epistemic_uncertainty,
        atol=1e-12,
    )
    for seed in [3, 7]:
        assert (run_dir / "members" / f"seed_{seed}" / "best.pt").exists()
        assert (run_dir / "members" / f"seed_{seed}" / "val_predictions.csv").exists()
    assert (run_dir / "val_member_predictions.npz").exists()
    metrics = evaluate_run(run_dir, torch.device('cpu'))
    assert metrics['n_members'] == 2
    assert "nll" in metrics
    assert "mean_epistemic_uncertainty" in metrics
    pd.testing.assert_frame_equal(predictions, pd.read_csv(run_dir / 'test_predictions.csv'))

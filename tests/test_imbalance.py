"""Imbalance handling (focal loss, balanced sampler) and saved validation predictions."""
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from src.losses import FocalLoss
from src.train import balanced_sampler

from .conftest import REPO


@pytest.mark.parametrize("weighted", [False, True])
def test_focal_loss_with_gamma_zero_is_cross_entropy(weighted):
    torch.manual_seed(0)
    logits, target = torch.randn(32, 6), torch.randint(0, 6, (32,))
    weight = torch.rand(6) + 0.1 if weighted else None
    expected = nn.CrossEntropyLoss(weight=weight)(logits, target)
    assert FocalLoss(gamma=0, weight=weight)(logits, target) == pytest.approx(expected.item())


def test_focal_loss_down_weights_easy_examples():
    target = torch.tensor([0, 0])
    easy, hard = torch.tensor([[5.0, 0, 0, 0, 0, 0]]), torch.tensor([[0.5, 0, 0, 0, 0, 0]])
    ce, focal = nn.CrossEntropyLoss(reduction="none"), FocalLoss(gamma=2)
    ratio_easy = focal(easy, target[:1]) / ce(easy, target[:1])
    ratio_hard = focal(hard, target[:1]) / ce(hard, target[:1])
    assert ratio_easy < ratio_hard < 1
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1)


def test_balanced_sampler_draws_classes_equally_often():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 9 + [3, 4, 5])
    drawn = labels[list(balanced_sampler(labels, seed=0))]
    assert len(drawn) == len(labels)
    shares = np.bincount(drawn, minlength=6) / len(drawn)
    np.testing.assert_allclose(shares, 1 / 6, atol=0.04)
    assert list(balanced_sampler(labels, seed=0)) == list(balanced_sampler(labels, seed=0))
    with pytest.raises(ValueError):
        balanced_sampler(np.array([0, 1, 2]), seed=0)


@pytest.mark.parametrize("overrides", [
    ["experiment=deep_ensemble_focal"],
    ["experiment=deep_ensemble", "train.balanced_sampler=true", "name=sampler"],
])
def test_imbalance_ensembles_train_and_save_validation(overrides, synthetic_root, tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "src.train", *overrides,
        "ensemble.seeds=[3,7]", f"data.root={synthetic_root}",
        f"output_root={tmp_path / 'outputs'}", "data.image_size=32",
        "train.epochs=1", "train.batch_size=16", "device=cpu",
        "model.image_encoder.pretrained=false", "model.image_encoder.freeze=false",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    run_dir = next((tmp_path / "outputs" / "runs").glob("*-patient_holdout-f0-s42"))
    with np.load(run_dir / "test_member_predictions.npz", allow_pickle=False) as test, \
            np.load(run_dir / "val_member_predictions.npz", allow_pickle=False) as val:
        assert val["probabilities"].shape[0] == 2
        assert val["probabilities"].shape[1] == len(val["labels"]) > 0
        np.testing.assert_allclose(val["probabilities"].sum(axis=-1), 1, atol=1e-5)
        assert not set(val["patient_ids"]) & set(test["patient_ids"])
    for seed in (3, 7):
        assert (run_dir / "members" / f"seed_{seed}" / "val_predictions.csv").is_file()


def test_analysis_runs_on_real_pipeline_outputs(synthetic_root, tmp_path):
    """Both arms through src.train, then src.analysis with temperature scaling."""
    outputs = tmp_path / "outputs"
    common = [
        "ensemble.seeds=[3,7]", f"data.root={synthetic_root}", f"output_root={outputs}",
        "data.image_size=32", "train.epochs=1", "train.batch_size=16", "device=cpu",
        "model.image_encoder.pretrained=false", "model.image_encoder.freeze=false",
    ]
    for experiment in ("deep_ensemble_focal", "deep_ensemble_image_only_focal"):
        result = subprocess.run([sys.executable, "-m", "src.train",
                                 f"experiment={experiment}", *common],
                                cwd=REPO, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run([
        sys.executable, "-m", "src.analysis",
        "deep_ensemble_image_only_focal:deep_ensemble_focal",
        "--runs-root", str(outputs / "runs"), "--out", str(tmp_path / "analysis"),
        "--n-boot", "50",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "After temperature scaling" in (tmp_path / "analysis" / "report.md").read_text()

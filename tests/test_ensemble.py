"""Entropy limits and saved ensemble pipeline round trips."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluate import evaluate_run
from src.uq.deep_ensemble import DeepEnsemble, uncertainty

from .conftest import REPO


@pytest.mark.parametrize("seeds", [[0], [0, 0], [-1, 2], [0, 1.5]])
def test_invalid_seeds(seeds):
    with pytest.raises(ValueError):
        DeepEnsemble(seeds)


def test_entropy_decomposition():
    total, aleatoric, epistemic = uncertainty(np.array([[[1., 0.]], [[0., 1.]]]))
    np.testing.assert_allclose(total, np.log(2))
    np.testing.assert_allclose(aleatoric, 0)
    np.testing.assert_allclose(epistemic, total)
    total, aleatoric, epistemic = uncertainty(np.full((2, 3, 2), .5))
    np.testing.assert_allclose(total, aleatoric)
    np.testing.assert_allclose(epistemic, 0, atol=1e-12)


@pytest.mark.parametrize("experiment", ["deep_ensemble", "deep_ensemble_image_only"])
@pytest.mark.parametrize("split", ["patient_holdout", "patient_kfold"])
def test_ensemble_pipeline(experiment, split, synthetic_root, tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "src.train", f"experiment={experiment}",
        "ensemble.seeds=[3,7]", f"data.root={synthetic_root}",
        f"output_root={tmp_path / 'outputs'}", "data.image_size=32",
        "train.epochs=1", "train.batch_size=16", "device=cpu",
        "model.image_encoder.pretrained=false", "model.image_encoder.freeze=false",
        f"split={split}",
    ], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / 'outputs' / 'runs' / f'{experiment}-{split}-f0-s42'
    predictions = pd.read_csv(run_dir / 'test_predictions.csv')
    with np.load(run_dir / 'test_member_predictions.npz', allow_pickle=False) as saved:
        assert saved['probabilities'].shape == (2, len(predictions), 6)
        np.testing.assert_array_equal(saved['seeds'], [3, 7])
        np.testing.assert_allclose(saved['probabilities'].sum(axis=-1), 1, atol=1e-6)
        np.testing.assert_allclose(
            saved['probabilities'].mean(axis=0),
            predictions[[f'p_{c}' for c in saved['classes']]],
        )
    for partition in ('val', 'test'):
        with np.load(run_dir / f'{partition}_member_predictions.npz', allow_pickle=False) as saved:
            logits = torch.from_numpy(saved['logits'])
            np.testing.assert_allclose(torch.softmax(logits, dim=-1).numpy(),
                                       saved['probabilities'], rtol=1e-6)
    for seed in [3, 7]:
        member = run_dir / 'members' / f'seed_{seed}'
        for artifact in ('split_metadata.csv', 'metadata_missingness.csv',
                         'metadata_preprocessor.joblib',
                         'provenance.json', 'image_hashes.json', 'source_snapshot/src/train.py'):
            assert (member / artifact).exists()
        rows = pd.read_csv(member / 'split_metadata.csv')
        assert rows.groupby('patient_id').partition.nunique().max() == 1
    np.testing.assert_allclose(
        predictions.total_uncertainty,
        predictions.aleatoric_uncertainty + predictions.epistemic_uncertainty,
        atol=1e-12,
    )
    for seed in [3, 7]:
        assert (run_dir / 'members' / f'seed_{seed}' / 'best.pt').exists()
    # New runs must not depend on the mutable source metadata during re-evaluation.
    (synthetic_root / 'metadata.csv').unlink()
    metrics = evaluate_run(run_dir, torch.device('cpu'))
    assert metrics['n_members'] == 2
    pd.testing.assert_frame_equal(predictions, pd.read_csv(run_dir / 'test_predictions.csv'))

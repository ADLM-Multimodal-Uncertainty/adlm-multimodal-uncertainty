"""Independent seeds on a shared split; probability averaging and entropy in nats.

Members retain ordinary run artifacts. Evaluation rebuilds every member from its
saved configuration and checkpoint, loading one model at a time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import CLASSES
from src.evaluate import evaluate_run
from src.metrics import detailed_metrics


class DeepEnsemble:
    def __init__(self, seeds: list[int]):
        self.seeds = list(seeds)
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("deep ensembles require at least two distinct seeds")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")


def uncertainty(member_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predictive entropy, expected member entropy, and mutual information."""
    def entropy(p):
        return -(p * np.log(np.clip(p, np.finfo(float).tiny, 1))).sum(axis=-1)

    total = entropy(member_probs.mean(axis=0))
    aleatoric = entropy(member_probs).mean(axis=0)
    return total, aleatoric, np.maximum(total - aleatoric, 0)


def evaluate_ensemble(run_dir: Path, device: torch.device) -> dict:
    import json

    cfg = OmegaConf.load(run_dir / "config.yaml")
    ensemble = instantiate(cfg.ensemble)
    frames, epochs = [], []
    for seed in ensemble.seeds:
        member_dir = run_dir / "members" / f"seed_{seed}"
        metrics = evaluate_run(member_dir, device)
        frame = pd.read_csv(member_dir / "test_predictions.csv", dtype={
            "patient_id": str, "img_id": str,
        })
        if frames and not frame[["patient_id", "img_id", "diagnostic"]].equals(
            frames[0][["patient_id", "img_id", "diagnostic"]]
        ):
            raise ValueError("ensemble members must use identical ordered test samples")
        frames.append(frame)
        epochs.append(metrics["best_epoch"])

    for partition in ("val", "test"):
        archives = []
        for seed in ensemble.seeds:
            with np.load(run_dir / "members" / f"seed_{seed}" /
                         f"{partition}_predictions.npz", allow_pickle=False) as archive:
                archives.append({key: archive[key] for key in archive.files})
        for archive in archives[1:]:
            for key in ("img_ids", "patient_ids", "labels", "classes"):
                if not np.array_equal(archive[key], archives[0][key]):
                    raise ValueError(f"ensemble {partition} {key} mismatch")
        np.savez_compressed(
            run_dir / f"{partition}_member_predictions.npz",
            probabilities=np.stack([a["probabilities"] for a in archives]),
            logits=np.stack([a["logits"] for a in archives]),
            seeds=np.asarray(ensemble.seeds),
            **{key: archives[0][key] for key in
               ("img_ids", "patient_ids", "labels", "classes")},
        )

    columns = [f"p_{c}" for c in CLASSES]
    member_probs = np.stack([frame[columns].to_numpy() for frame in frames])
    probs = member_probs.mean(axis=0)
    total, aleatoric, epistemic = uncertainty(member_probs)
    predictions = frames[0].copy()
    predictions[columns] = probs
    predictions["predicted"] = [CLASSES[i] for i in probs.argmax(axis=1)]
    predictions["total_uncertainty"] = total
    predictions["aleatoric_uncertainty"] = aleatoric
    predictions["epistemic_uncertainty"] = epistemic
    labels = predictions["diagnostic"].map(CLASSES.index).to_numpy()
    metrics = {
        **detailed_metrics(labels, probs.argmax(axis=1), CLASSES),
        "n_images": len(labels),
        "n_patients": int(predictions.patient_id.nunique()),
        "n_members": len(ensemble.seeds),
        "member_seeds": ensemble.seeds,
        "member_best_epochs": epochs,
    }
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

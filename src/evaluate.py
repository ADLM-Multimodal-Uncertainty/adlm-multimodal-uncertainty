"""Evaluate a checkpoint on the validation and test partitions of its saved split.

    python -m src.evaluate outputs/runs/<run_group>/fold_<k>

`src.train` calls this automatically. The fold, including the metadata preprocessor,
is rebuilt from the run's saved config. An ensemble directory aggregates its members;
pass ``reuse_member_predictions`` only immediately after those members were evaluated
in the same process, so training does not run the test set twice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src import CLASSES
from src.data.dataset import SkinLesionDataset, build_fold
from src.metrics import detailed_metrics, probability_metrics
from src.models import MultimodalClassifier, build_model
from src.uq.deep_ensemble import uncertainty

SPLITS = ("val", "test")


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Predictions:
    probs: np.ndarray  # (N, K)
    labels: np.ndarray  # (N,)
    loss: float | None  # mean `criterion` over the loader, None without a criterion


@torch.no_grad()
def predict(
    model: MultimodalClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Predictions:
    """Probabilities, labels and optionally the loss. Does not change the train/eval mode."""
    probs, labels = [], []
    loss_sum = 0.0
    for batch in loader:
        meta = batch["meta"].to(device) if model.uses_metadata else None
        logits = model(batch["image"].to(device), meta).float()
        if criterion is not None:
            loss_sum += criterion(logits, batch["label"].to(device)).item() * len(batch["label"])
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
    labels = np.concatenate(labels)
    loss = loss_sum / len(labels) if criterion is not None else None
    return Predictions(np.concatenate(probs), labels, loss)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _prediction_frame(dataset: SkinLesionDataset, probs: np.ndarray) -> pd.DataFrame:
    preds = probs.argmax(axis=1)
    frame = dataset.df[["patient_id", "img_id", "diagnostic"]].copy()
    frame["predicted"] = [CLASSES[i] for i in preds]
    for index, name in enumerate(CLASSES):
        frame[f"p_{name}"] = probs[:, index]
    return frame


def _split_metrics(
    dataset: SkinLesionDataset, probs: np.ndarray, best_epoch: int, split: str
) -> dict:
    labels = dataset.labels
    preds = probs.argmax(axis=1)
    return {
        **detailed_metrics(labels, preds, CLASSES),
        **probability_metrics(labels, probs),
        "best_epoch": best_epoch,
        "split": split,
        "n_images": len(labels),
        "n_patients": int(dataset.df["patient_id"].nunique()),
    }


def evaluate_single_run(run_dir: Path, device: torch.device) -> dict:
    """Score one checkpoint on validation and test. Returns the test metrics."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    data = build_fold(cfg.data, instantiate(cfg.split), cfg.fold)
    model = build_model(cfg.model, len(CLASSES), data.n_meta_features)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    results = {}
    for split in SPLITS:
        dataset = getattr(data, split)
        loader = DataLoader(
            dataset, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers
        )
        probs = predict(model, loader, device).probs
        metrics = _split_metrics(dataset, probs, int(state["epoch"]), split)
        _write_json(run_dir / f"{split}_metrics.json", metrics)
        _prediction_frame(dataset, probs).to_csv(run_dir / f"{split}_predictions.csv", index=False)
        results[split] = metrics
    return results["test"]


def _member_predictions_ready(member_dir: Path) -> bool:
    return all(
        (member_dir / f"{split}_predictions.csv").exists()
        and (member_dir / f"{split}_metrics.json").exists()
        for split in SPLITS
    )


def _aggregate_split(
    frames: list[pd.DataFrame], epochs: list[int], seeds: list[int], split: str
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    columns = [f"p_{name}" for name in CLASSES]
    member_probs = np.stack([frame[columns].to_numpy(dtype=np.float64) for frame in frames])
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
        **probability_metrics(labels, probs),
        "split": split,
        "n_images": len(labels),
        "n_patients": int(predictions.patient_id.nunique()),
        "n_members": len(seeds),
        "member_seeds": seeds,
        "member_best_epochs": epochs,
        "mean_total_uncertainty": float(total.mean()),
        "mean_aleatoric_uncertainty": float(aleatoric.mean()),
        "mean_epistemic_uncertainty": float(epistemic.mean()),
    }
    return metrics, predictions, member_probs


def evaluate_ensemble(
    run_dir: Path, device: torch.device, reuse_member_predictions: bool = False
) -> dict:
    """Average member probabilities on validation and test, and save both."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    ensemble = instantiate(cfg.ensemble)
    seeds = list(ensemble.seeds)
    by_split: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
    epochs: list[int] = []
    for seed in seeds:
        member_dir = run_dir / "members" / f"seed_{seed}"
        if not (reuse_member_predictions and _member_predictions_ready(member_dir)):
            evaluate_single_run(member_dir, device)
        epochs.append(json.loads((member_dir / "test_metrics.json").read_text())["best_epoch"])
        for split in SPLITS:
            frame = pd.read_csv(
                member_dir / f"{split}_predictions.csv",
                dtype={"patient_id": str, "img_id": str},
            )
            if by_split[split] and not frame[["patient_id", "img_id", "diagnostic"]].equals(
                by_split[split][0][["patient_id", "img_id", "diagnostic"]]
            ):
                raise ValueError("ensemble members must use identical ordered samples")
            by_split[split].append(frame)

    test_metrics = {}
    for split in SPLITS:
        metrics, predictions, member_probs = _aggregate_split(by_split[split], epochs, seeds, split)
        predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
        labels = predictions["diagnostic"].map(CLASSES.index).to_numpy()
        np.savez_compressed(
            run_dir / f"{split}_member_predictions.npz",
            probabilities=member_probs,
            labels=labels,
            img_ids=predictions.img_id.to_numpy(dtype=str),
            patient_ids=predictions.patient_id.to_numpy(dtype=str),
            classes=np.asarray(CLASSES),
            seeds=np.asarray(seeds),
        )
        _write_json(run_dir / f"{split}_metrics.json", metrics)
        test_metrics = metrics if split == "test" else test_metrics
    return test_metrics


def evaluate_run(run_dir: Path, device: torch.device) -> dict:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if cfg.get("ensemble") is not None:
        return evaluate_ensemble(run_dir, device)
    return evaluate_single_run(run_dir, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate_run(args.run_dir, resolve_device(args.device))
    shown = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier")
    print(json.dumps({key: metrics[key] for key in shown}))


if __name__ == "__main__":
    main()

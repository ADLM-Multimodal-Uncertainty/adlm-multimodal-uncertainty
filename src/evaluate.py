"""Evaluate a run's best checkpoint once on its test partition.

    python -m src.evaluate outputs/runs/<run_name>

`src.train` calls this automatically after training. The fold (including the metadata
preprocessor) is rebuilt from the run's saved config instead of being pickled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src import CLASSES
from src.data.dataset import build_fold
from src.metrics import detailed_metrics
from src.models import MultimodalClassifier, build_model


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


def evaluate_run(run_dir: Path, device: torch.device) -> dict:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    data = build_fold(cfg.data, instantiate(cfg.split), cfg.fold)
    model = build_model(cfg.model, len(CLASSES), data.n_meta_features)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    loader = DataLoader(
        data.test, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers
    )
    prediction = predict(model, loader, device)
    probs, labels = prediction.probs, prediction.labels
    preds = probs.argmax(axis=1)
    test_df = data.test.df
    metrics = {
        **detailed_metrics(labels, preds, CLASSES),
        "best_epoch": state["epoch"],
        "n_images": len(labels),
        "n_patients": int(test_df["patient_id"].nunique()),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    predictions = test_df[["patient_id", "img_id", "diagnostic"]].copy()
    predictions["predicted"] = [CLASSES[i] for i in preds]
    for k, name in enumerate(CLASSES):
        predictions[f"p_{name}"] = probs[:, k]
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics = evaluate_run(args.run_dir, resolve_device(args.device))
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}))


if __name__ == "__main__":
    main()

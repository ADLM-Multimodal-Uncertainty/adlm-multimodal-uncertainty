"""Train one configuration on one fold with one seed, then evaluate it on the test partition.

    scripts/local/run.sh experiment=draft_baseline
    scripts/local/run.sh experiment=draft_image_only data.root=/path/to/pad_ufes_20

Writes to `run_dir`: the resolved `config.yaml`, `history.csv`, `best.pt` (best epoch by
`train.monitor`), `test_metrics.json` and `test_predictions.csv`.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from src import CLASSES
from src.data.dataset import build_fold
from src.evaluate import evaluate_run, predict, resolve_device
from src.metrics import summary_metrics
from src.models import MultimodalClassifier, build_model

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES))
    if (counts == 0).any():
        raise ValueError("weighted_loss needs every class in the training partition")
    return torch.tensor(len(labels) / (len(CLASSES) * counts), dtype=torch.float32, device=device)


def train_epoch(
    model: MultimodalClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        label = batch["label"].to(device)
        meta = batch["meta"].to(device) if model.uses_metadata else None
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch["image"].to(device), meta), label)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(label)
        n += len(label)
    return total / n


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    log.info("run %s on %s", cfg.run_name, device)

    data = build_fold(cfg.data, instantiate(cfg.split), cfg.fold)
    model = build_model(cfg.model, len(CLASSES), data.n_meta_features).to(device)
    train_loader = DataLoader(
        data.train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        data.val, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers
    )
    optimizer = instantiate(
        cfg.train.optimizer, params=[p for p in model.parameters() if p.requires_grad]
    )
    weight = class_weights(data.train.labels, device) if cfg.train.weighted_loss else None
    criterion = instantiate(cfg.train.loss, weight=weight)

    monitor = cfg.train.monitor
    best = -np.inf
    with (run_dir / "history.csv").open("w", newline="") as f:
        fields = ["epoch", "train_loss", "val_accuracy", "val_balanced_accuracy", "val_macro_f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            model.eval()
            probs, labels = predict(model, val_loader, device)
            val = summary_metrics(labels, probs.argmax(axis=1), len(CLASSES))
            record = {"epoch": epoch, "train_loss": train_loss}
            record.update({f"val_{k}": v for k, v in val.items()})
            writer.writerow(record)
            f.flush()
            log.info(json.dumps(record))
            if record[monitor] > best:
                best = record[monitor]
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, monitor: best},
                    run_dir / "best.pt",
                )

    metrics = evaluate_run(run_dir, device)
    log.info(
        "best %s %.4f; test accuracy %.4f, balanced accuracy %.4f, macro-F1 %.4f",
        monitor, best, metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"],
    )


if __name__ == "__main__":
    main()

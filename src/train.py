"""Train one configuration on one fold, then evaluate validation and test.

    scripts/local/run.sh experiment=baseline_multimodal
    scripts/local/run.sh -m experiment=baseline_image_only

Writes the resolved config, `history.csv`, `best.pt`, and validation and test
predictions. Ensemble runs repeat that once per member, then write the averaged
predictions and uncertainty columns to the fold directory.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src import CLASSES, sweep_tag
from src.data.dataset import build_fold
from src.evaluate import Predictions, evaluate_ensemble, evaluate_run, predict, resolve_device
from src.metrics import METRIC_NAMES, probability_metrics, summary_metrics
from src.models import MultimodalClassifier, build_model
from src.visualize import plot_run

log = logging.getLogger(__name__)

MINIMIZE_MONITORS = {"val_loss", "val_nll", "val_brier"}
MAXIMIZE_MONITORS = {"val_accuracy", "val_balanced_accuracy", "val_macro_f1"}


class NoLearningRateScheduler:
    """Constant learning rate. Exists so the scheduler config group has an explicit option."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels: np.ndarray, device: torch.device, power: float) -> torch.Tensor | None:
    """Inverse-frequency weights raised to ``power`` and rescaled to mean 1.

    ``power=0`` is unweighted and returns None. ``power=1`` keeps the ratios
    ``N / (K n_k)``. Dividing by the mean keeps the loss on a similar scale when
    the power changes, which makes validation loss comparable across that sweep.
    """
    if power == 0:
        return None
    if power < 0:
        raise ValueError("class_weight_power must be non-negative")
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    if (counts == 0).any():
        missing = [CLASSES[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(
            "weighted_loss needs every class in the training partition; "
            "missing " + ", ".join(missing)
        )
    weights = (len(labels) / (len(CLASSES) * counts)) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def monitor_direction(name: str) -> str:
    if name in MINIMIZE_MONITORS:
        return "min"
    if name in MAXIMIZE_MONITORS:
        return "max"
    known = ", ".join(sorted(MINIMIZE_MONITORS | MAXIMIZE_MONITORS))
    raise ValueError(f"unknown monitor {name!r}; expected one of {known}")


def improved(value: float, best: float, direction: str, min_delta: float) -> bool:
    """The first finite score beats the +/- infinity sentinel used before epoch 1."""
    if not np.isfinite(value):
        return False
    if not np.isfinite(best):
        return True
    if direction == "min":
        return value < best - min_delta
    return value > best + min_delta


def drop_last_batch(n_examples: int, batch_size: int, fine_tune: bool) -> bool:
    """Drop a final batch of size 1. BatchNorm cannot update on a single image."""
    if not fine_tune or n_examples <= batch_size:
        return False
    return n_examples % batch_size == 1


def score(prediction: Predictions, prefix: str) -> dict[str, float]:
    metrics = summary_metrics(prediction.labels, prediction.probs.argmax(axis=1), len(CLASSES))
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def apply_multirun_directory(cfg: DictConfig) -> None:
    """If the run directory was resolved before the sweep tag, move this job aside.

    A user-supplied ``tag`` is left unchanged. When ``${sweep_tag:}`` already returned
    the same suffix, the directory is already unique and this is a no-op.
    """
    suffix = sweep_tag()
    if not suffix or cfg.get("tag") not in ("", None):
        return
    cfg.tag = suffix
    cfg.run_group = f"{cfg.name}{suffix}"
    cfg.run_name = f"{cfg.run_group}/fold_{cfg.fold}"
    cfg.run_dir = str(Path(cfg.output_root) / "runs" / cfg.run_name)


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


def _build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer, monitor: str):
    scheduler_cfg = cfg.train.get("scheduler")
    if scheduler_cfg is None:
        return None
    target = str(scheduler_cfg.get("_target_", ""))
    if target.endswith("NoLearningRateScheduler"):
        return None
    scheduler = instantiate(scheduler_cfg, optimizer=optimizer)
    if isinstance(scheduler, NoLearningRateScheduler):
        return None
    if isinstance(scheduler, ReduceLROnPlateau):
        direction = monitor_direction(monitor)
        if scheduler.mode != direction:
            raise ValueError(
                f"ReduceLROnPlateau mode={scheduler.mode!r} does not match "
                f"monitor {monitor!r} ({direction})"
            )
    return scheduler


def train_run(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    log.info("run %s on %s", cfg.run_name, device)

    data = build_fold(cfg.data, instantiate(cfg.split), cfg.fold)
    model = build_model(cfg.model, len(CLASSES), data.n_meta_features).to(device)
    fine_tune = not bool(cfg.model.image_encoder.freeze)
    if fine_tune and len(data.train) == 1:
        raise ValueError("fine-tuning BatchNorm needs more than one training image")
    train_loader = DataLoader(
        data.train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=drop_last_batch(len(data.train), cfg.train.batch_size, fine_tune),
    )
    val_loader = DataLoader(
        data.val, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers
    )
    optimizer = instantiate(
        cfg.train.optimizer, params=[p for p in model.parameters() if p.requires_grad]
    )
    power = float(cfg.train.class_weight_power) if cfg.train.weighted_loss else 0.0
    weight = class_weights(data.train.labels, device, power)
    criterion = instantiate(cfg.train.loss, weight=weight)
    monitor = str(cfg.train.monitor)
    direction = monitor_direction(monitor)
    scheduler = _build_scheduler(cfg, optimizer, monitor)

    train_eval_loader = None
    if cfg.train.track_train_metrics:
        train_eval_loader = DataLoader(
            data.train, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers
        )

    best = np.inf if direction == "min" else -np.inf
    stale = 0
    min_delta = float(cfg.train.early_stopping_min_delta)
    patience = cfg.train.get("early_stopping_patience")
    patience = None if patience is None else int(patience)
    with (run_dir / "history.csv").open("w", newline="") as handle:
        fields = ["epoch", "lr", "train_loss", "val_loss", "val_nll", "val_brier"]
        if train_eval_loader is not None:
            fields += [f"train_{name}" for name in METRIC_NAMES]
        fields += [f"val_{name}" for name in METRIC_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, int(cfg.train.epochs) + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            model.eval()
            val = predict(model, val_loader, device, criterion)
            probabilities = probability_metrics(val.labels, val.probs)
            record = {
                "epoch": epoch,
                "lr": learning_rate,
                "train_loss": train_loss,
                "val_loss": val.loss,
                "val_nll": probabilities["nll"],
                "val_brier": probabilities["brier"],
            }
            if train_eval_loader is not None:
                # The training transform draws random numbers; restore the generator so
                # tracking train metrics does not change the batches or dropout of later epochs.
                with torch.random.fork_rng(devices=[]):
                    record.update(score(predict(model, train_eval_loader, device), "train"))
            record.update(score(val, "val"))
            writer.writerow(record)
            handle.flush()
            log.info(json.dumps(record))

            value = float(record[monitor])
            first_checkpoint = not (run_dir / "best.pt").exists()
            if first_checkpoint or improved(value, best, direction, min_delta):
                if np.isfinite(value):
                    best = value
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch, monitor: best},
                    run_dir / "best.pt",
                )
                stale = 0
            else:
                stale += 1
            if scheduler is not None and np.isfinite(value):
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(value)
                else:
                    scheduler.step()
            if patience is not None and stale >= patience:
                log.info("early stopping at epoch %s (best %s %.6f)", epoch, monitor, best)
                break

    metrics = evaluate_run(run_dir, device)
    plot_run(run_dir)
    log.info(
        "best %s %.4f; test accuracy %.4f, balanced accuracy %.4f, macro-F1 %.4f, NLL %.4f",
        monitor,
        best,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["nll"],
    )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    apply_multirun_directory(cfg)
    if cfg.get("ensemble") is None:
        train_run(cfg)
        return
    ensemble = instantiate(cfg.ensemble)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    for seed in ensemble.seeds:
        member = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        member.ensemble = None
        member.seed = seed
        member.run_name = f"{cfg.run_name}/members/seed_{seed}"
        member.run_dir = str(run_dir / "members" / f"seed_{seed}")
        train_run(member)
    evaluate_ensemble(run_dir, resolve_device(cfg.device), reuse_member_predictions=True)


if __name__ == "__main__":
    main()

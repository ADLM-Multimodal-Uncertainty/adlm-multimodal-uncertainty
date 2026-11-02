"""Plots of a finished run, drawn from its `history.csv` and `test_metrics.json`.

    python -m src.visualize outputs/runs/<run_name>

`src.train` calls this after evaluation. Writes `curves.png` (training loss and validation
metrics per epoch, best epoch marked) and `confusion_matrix.png` (test partition).
Uses `Figure` directly instead of pyplot, so it needs no display or backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# (panel title, y limits, [(history column suffix, legend label)])
PANELS = [
    ("loss", None, [("loss", "loss")]),
    ("macro-F1", (0, 1), [("macro_f1", "macro-F1")]),
    ("accuracy", (0, 1), [("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy")]),
]


def plot_curves(history: pd.DataFrame, best_epoch: int | None, title: str, path: Path) -> None:
    """Training (dashed) against validation (solid) per epoch. Columns missing from older
    histories (validation loss, training metrics) are skipped. A growing gap between the two
    lines, or a validation loss that turns upward, indicates overfitting."""
    fig = Figure(figsize=(15, 4), layout="constrained")
    for ax, (panel, ylim, series) in zip(fig.subplots(1, len(PANELS)), PANELS, strict=True):
        for i, (suffix, label) in enumerate(series):
            for split, style in (("train", "--"), ("val", "-")):
                column = f"{split}_{suffix}"
                if column in history:
                    ax.plot(
                        history["epoch"], history[column], style, marker="o", color=f"C{i}",
                        label=f"{split} {label}",
                    )
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle=":")
        ax.set(xlabel="epoch", title=panel, ylim=ylim)
        ax.set_xticks(history["epoch"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if best_epoch is not None:
        title += f" (dotted line: best epoch {best_epoch})"
    fig.suptitle(title)
    fig.savefig(path, dpi=150)


def plot_confusion_matrix(
    confusion: np.ndarray, classes: list[str], title: str, path: Path
) -> None:
    """Colour is the fraction of each true class (row); the number is the image count."""
    support = confusion.sum(axis=1, keepdims=True)
    fraction = np.divide(confusion, support, out=np.zeros(confusion.shape), where=support > 0)
    fig = Figure(figsize=(6.5, 5.5), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(image, ax=ax, label="fraction of true class")
    ticks = range(len(classes))
    ax.set_xticks(ticks, classes)
    ax.set_yticks(ticks, [f"{c} (n={n})" for c, n in zip(classes, support[:, 0], strict=True)])
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in ticks:
        for j in ticks:
            colour = "white" if fraction[i, j] > 0.5 else "black"
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color=colour)
    fig.savefig(path, dpi=150)


def plot_run(run_dir: Path) -> list[Path]:
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    history = pd.read_csv(run_dir / "history.csv")
    curves, confusion = run_dir / "curves.png", run_dir / "confusion_matrix.png"
    plot_curves(history, metrics.get("best_epoch"), run_dir.name, curves)
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        metrics["class_order"],
        f"{run_dir.name}\ntest set, {metrics['n_images']} images",
        confusion,
    )
    return [curves, confusion]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for path in plot_run(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()

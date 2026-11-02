import json

import numpy as np
import pandas as pd

from src.visualize import plot_run

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


OLD_HISTORY = {
    "epoch": [1, 2, 3],
    "train_loss": [1.2, 0.8, 0.6],
    "val_accuracy": [0.5, 0.6, 0.7],
    "val_balanced_accuracy": [0.3, 0.4, 0.5],
    "val_macro_f1": [0.2, 0.35, 0.45],
}
FULL_HISTORY = {
    **OLD_HISTORY,
    "val_loss": [1.3, 1.0, 1.1],
    "train_accuracy": [0.6, 0.7, 0.8],
    "train_balanced_accuracy": [0.4, 0.5, 0.6],
    "train_macro_f1": [0.3, 0.5, 0.7],
}


def _write_run(run_dir, confusion, classes, with_best_epoch=True, history=FULL_HISTORY):
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    metrics = {
        "confusion_matrix": confusion.tolist(),
        "class_order": classes,
        "n_images": int(confusion.sum()),
    }
    if with_best_epoch:
        metrics["best_epoch"] = 3
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics))


def test_plot_run_writes_both_pngs(tmp_path):
    classes = ["A", "B", "C"]
    _write_run(tmp_path, np.array([[5, 1, 0], [2, 3, 0], [0, 1, 4]]), classes)
    paths = plot_run(tmp_path)
    assert [p.name for p in paths] == ["curves.png", "confusion_matrix.png"]
    for path in paths:
        assert path.read_bytes().startswith(PNG_MAGIC)


def test_class_without_test_images_does_not_break_the_confusion_matrix(tmp_path):
    # An all-zero row has no support; normalising it must not divide by zero.
    classes = ["A", "B", "C"]
    _write_run(tmp_path, np.array([[5, 1, 0], [0, 0, 0], [0, 1, 4]]), classes)
    for path in plot_run(tmp_path):
        assert path.exists()


def test_history_without_best_epoch_still_plots(tmp_path):
    _write_run(tmp_path, np.eye(3, dtype=int), ["A", "B", "C"], with_best_epoch=False)
    assert all(path.exists() for path in plot_run(tmp_path))


def test_history_from_before_validation_loss_was_tracked_still_plots(tmp_path):
    _write_run(tmp_path, np.eye(3, dtype=int), ["A", "B", "C"], history=OLD_HISTORY)
    assert all(path.exists() for path in plot_run(tmp_path))

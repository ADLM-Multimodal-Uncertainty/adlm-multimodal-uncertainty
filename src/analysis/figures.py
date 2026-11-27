"""Figures for one result-analysis directory.

F1, F2, and the dEU candles are written once per comparison. F3, F4, and F6 summarize
the experiment. F5 is the cross-experiment scatter and is written from the summary table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.analysis.effects import quadrant_counts

QUADRANTS = (
    ("both_correct", "Both correct", "Both correct"),
    ("reference_wrong_candidate_correct", "Reference wrong, candidate correct", "Fixed"),
    ("reference_correct_candidate_wrong", "Reference correct, candidate wrong", "Broken"),
    ("both_wrong", "Both wrong", "Both wrong"),
)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return path


def plot_paired_eu_scatter(paired: pd.DataFrame, title: str, badge: str, path: Path) -> Path:
    reference = paired["reference_epistemic"].to_numpy()
    candidate = paired["candidate_epistemic"].to_numpy()
    limit = max(float(np.max(reference)), float(np.max(candidate)), 1e-3) * 1.05
    fig = Figure(figsize=(5.5, 5.2), layout="constrained")
    ax = fig.subplots()
    ax.scatter(reference, candidate, s=12, alpha=0.45, linewidths=0)
    ax.plot([0, limit], [0, limit], color="black", linewidth=1)
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Reference epistemic uncertainty (nats)",
        ylabel="Candidate epistemic uncertainty (nats)",
        title=f"{title}\n{badge}, {len(paired)} images",
        aspect="equal",
    )
    ax.grid(alpha=0.3)
    return _save(fig, path)


def _quadrant_samples(paired: pd.DataFrame) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    reference_correct = paired["reference_correct"].to_numpy(dtype=bool)
    candidate_correct = paired["candidate_correct"].to_numpy(dtype=bool)
    counts = dict(quadrant_counts(reference_correct, candidate_correct))
    masks = {
        "both_correct": reference_correct & candidate_correct,
        "reference_wrong_candidate_correct": ~reference_correct & candidate_correct,
        "reference_correct_candidate_wrong": reference_correct & ~candidate_correct,
        "both_wrong": ~reference_correct & ~candidate_correct,
    }
    values = paired["epistemic_reduction"].to_numpy(dtype=np.float64)
    samples = {key: values[mask][np.isfinite(values[mask])] for key, mask in masks.items()}
    return counts, samples


def plot_deu_distribution(paired: pd.DataFrame, title: str, badge: str, path: Path) -> Path:
    counts, samples = _quadrant_samples(paired)
    pieces = [sample for sample in samples.values() if len(sample)]
    finite = np.concatenate(pieces) if pieces else np.array([])
    if len(finite) == 0:
        edges = np.linspace(-1, 1, 21)
    else:
        span = max(float(np.max(np.abs(finite))), 1e-3)
        edges = np.linspace(-span, span, 21)
    fig = Figure(figsize=(12.5, 3.6), layout="constrained")
    axes = fig.subplots(1, 4, sharex=True, sharey=True)
    peak = 1
    histograms = []
    for key, _long_label, _short_label in QUADRANTS:
        hist, _ = np.histogram(samples[key], bins=edges)
        histograms.append(hist)
        peak = max(peak, int(hist.max()) if len(hist) else 1)
    for ax, (key, label, _short_label), hist in zip(axes, QUADRANTS, histograms, strict=True):
        ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color="C0")
        ax.axvline(0, color="black", linewidth=1)
        ax.set(title=f"{label}\n{counts[key]} images", xlabel="EU reduction (nats)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Images")
    axes[0].set_ylim(0, peak * 1.1)
    fig.suptitle(f"{title}\n{badge}")
    return _save(fig, path)


def plot_deu_candles(paired: pd.DataFrame, title: str, badge: str, path: Path) -> Path:
    """Box-and-whisker candles of dEU for the four correctness groups.

    The box is the quartile range, the whiskers reach 1.5 IQR, and the diamond is the mean.
    Fixed means the candidate is correct where the reference is wrong. Broken is the reverse.
    """
    counts, samples = _quadrant_samples(paired)
    fig = Figure(figsize=(8.2, 5.2), layout="constrained")
    ax = fig.subplots()
    positions = []
    drawn = []
    for index, (key, _long_label, _short_label) in enumerate(QUADRANTS, start=1):
        if len(samples[key]) == 0:
            continue
        positions.append(index)
        drawn.append(samples[key])
    if drawn:
        plotted = ax.boxplot(
            drawn,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "alpha": 0.35,
                "markeredgewidth": 0,
                "markerfacecolor": "C0",
            },
        )
        for patch in plotted["boxes"]:
            patch.set_facecolor("C0")
            patch.set_alpha(0.8)
        ax.scatter(
            positions,
            [float(np.mean(sample)) for sample in drawn],
            marker="D",
            s=22,
            color="black",
            zorder=4,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(
        range(1, len(QUADRANTS) + 1),
        [f"{short}\n{counts[key]} images" for key, _long, short in QUADRANTS],
    )
    ax.set_xlabel(
        "Fixed: reference wrong, candidate correct.  "
        "Broken: reference correct, candidate wrong.\n"
        "Box: quartiles.  Whiskers: 1.5 IQR.  Diamond: mean.",
        fontsize=8,
    )
    ax.set(ylabel="EU reduction (nats)", title=f"{title}\n{badge}")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrices(
    matrices: list[tuple[str, np.ndarray]], classes: list[str], badge: str, path: Path
) -> Path:
    fig = Figure(figsize=(4.4 * len(matrices), 4.8), layout="constrained")
    axes = fig.subplots(1, len(matrices))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (label, matrix) in zip(axes, matrices, strict=True):
        support = matrix.sum(axis=1, keepdims=True)
        fraction = np.divide(matrix, support, out=np.zeros(matrix.shape), where=support > 0)
        image = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
        ticks = range(len(classes))
        ax.set_xticks(ticks, classes, rotation=45, ha="right")
        ax.set_yticks(ticks, classes)
        ax.set(xlabel="Predicted", ylabel="True diagnosis", title=label)
        for row in ticks:
            for column in ticks:
                colour = "white" if fraction[row, column] > 0.5 else "black"
                ax.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color=colour,
                )
    fig.colorbar(image, ax=axes, label="Fraction of true diagnosis", shrink=0.85)
    fig.suptitle(f"Confusion matrices\n{badge}")
    return _save(fig, path)


def plot_effect_estimates(rows: list[dict], badge: str, path: Path) -> Path:
    labels = [row["label"] for row in rows]
    means = np.array([row["mean"] for row in rows], dtype=np.float64)
    low = np.array([row["low"] for row in rows], dtype=np.float64)
    high = np.array([row["high"] for row in rows], dtype=np.float64)
    fig = Figure(figsize=(8, max(3.2, 0.38 * len(rows) + 1.2)), layout="constrained")
    ax = fig.subplots()
    positions = np.arange(len(rows))
    ax.barh(
        positions,
        means,
        xerr=[np.maximum(means - low, 0), np.maximum(high - means, 0)],
        color="C0",
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set(
        xlabel="Mean epistemic uncertainty reduction (nats)",
        title=(
            f"Paired epistemic uncertainty reduction\n{badge}; "
            "bars are 95% patient-cluster intervals"
        ),
    )
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_validation_curves(history: pd.DataFrame, path: Path) -> Path:
    setups = list(dict.fromkeys(history["setup"]))
    fig = Figure(figsize=(11, 4.4), layout="constrained")
    loss_ax, accuracy_ax = fig.subplots(1, 2, sharex=True)
    panels = (
        (loss_ax, "val_loss", "Validation loss"),
        (accuracy_ax, "val_accuracy_pct", "Validation accuracy (%)"),
    )
    for color, setup in enumerate(setups):
        subset = history[history["setup"] == setup]
        for (_, _), member in subset.groupby(["fold", "member_seed"], sort=False):
            for ax, column, _ in panels:
                ax.plot(
                    member["epoch"],
                    member[column],
                    color=f"C{color}",
                    alpha=0.25,
                    linewidth=1,
                )
        grouped = subset.groupby("epoch")
        mean = grouped[[column for _, column, _ in panels]].mean()
        counts = grouped.size()
        for ax, column, ylabel in panels:
            ax.plot(mean.index, mean[column], color=f"C{color}", linewidth=2, label=setup)
            ax.set(xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=0.3)
        previous = None
        for epoch, count in counts.items():
            if previous is None or count != previous:
                loss_ax.annotate(
                    f"{int(count)} members",
                    (epoch, mean.loc[epoch, "val_loss"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=f"C{color}",
                )
            previous = count
    loss_ax.legend(fontsize=8)
    fig.suptitle(
        "Validation curves\n"
        "Faint lines are members; the solid line is the mean of members still training"
    )
    return _save(fig, path)


def plot_total_vs_aleatoric(summary: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(7.2, 5.4), layout="constrained")
    ax = fig.subplots()
    for _, row in summary.iterrows():
        ax.scatter(row["mean_total_uncertainty_nats"], row["mean_aleatoric_uncertainty_nats"], s=40)
        ax.annotate(
            f"{row['experiment_id']}\n{row['setup']}",
            (row["mean_total_uncertainty_nats"], row["mean_aleatoric_uncertainty_nats"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set(
        xlabel="Mean total uncertainty (nats)",
        ylabel="Mean aleatoric uncertainty (nats)",
        title="Total versus aleatoric uncertainty\nOne point per completed test-split setup",
    )
    ax.grid(alpha=0.3)
    return _save(fig, path)

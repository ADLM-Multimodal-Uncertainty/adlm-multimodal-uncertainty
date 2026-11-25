"""Publication-ready matplotlib Figure builders for paired EU analysis.

Uses `Figure` directly instead of pyplot so plotting requires no display backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_uncertainty(frame: pd.DataFrame, label: str, path: Path) -> None:
    fig = Figure(figsize=(15, 4.2), layout="constrained")
    axes = fig.subplots(1, 3)
    axes[0].hist(frame["image_epistemic"], bins=30, alpha=0.6, label="image")
    axes[0].hist(frame["multimodal_epistemic"], bins=30, alpha=0.6, label="multimodal")
    axes[0].set(title="Epistemic uncertainty (MI)", xlabel="nats", ylabel="images")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(frame["depistemic"], bins=30, alpha=0.8)
    axes[1].axvline(0, color="black", linestyle=":")
    axes[1].axvline(
        frame["depistemic"].mean(), color="C1", linestyle="--", label="mean dEU"
    )
    axes[1].set(title="Paired dEU distribution", xlabel="EU(image) - EU(multimodal)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    class_names = sorted(frame["diagnostic"].unique())
    class_values = [
        frame.loc[frame["diagnostic"] == name, "depistemic"].to_numpy()
        for name in class_names
    ]
    axes[2].boxplot(class_values, tick_labels=class_names, showfliers=False)
    axes[2].axhline(0, color="black", linestyle=":")
    axes[2].set(title="dEU by diagnosis class", ylabel="nats")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"{label}: uncertainty distributions")
    fig.savefig(path, dpi=150)


def plot_paired_eu_scatter(
    frame: pd.DataFrame, patient: pd.DataFrame, label: str, path: Path
) -> None:
    fig = Figure(figsize=(16, 4.8), layout="constrained")
    axes = fig.subplots(1, 3)

    limit_img = float(
        max(frame["image_epistemic"].max(), frame["multimodal_epistemic"].max(), 1e-6)
    ) * 1.05
    for name, group in frame.groupby("diagnostic"):
        axes[0].scatter(
            group["image_epistemic"],
            group["multimodal_epistemic"],
            s=14,
            alpha=0.55,
            label=str(name),
        )
    axes[0].plot([0, limit_img], [0, limit_img], ":", color="black")
    axes[0].set(
        xlim=(0, limit_img),
        ylim=(0, limit_img),
        xlabel="EU(image) [nats]",
        ylabel="EU(image + metadata) [nats]",
        title="Per-image paired EU (below line = EU reduced)",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    limit_pat = float(
        max(patient["image_epistemic"].max(), patient["multimodal_epistemic"].max(), 1e-6)
    ) * 1.05
    frac_reduced = float((patient["depistemic"] > 0).mean())
    axes[1].scatter(
        patient["image_epistemic"],
        patient["multimodal_epistemic"],
        s=18,
        alpha=0.65,
        color="C0",
    )
    axes[1].plot([0, limit_pat], [0, limit_pat], ":", color="black")
    axes[1].set(
        xlim=(0, limit_pat),
        ylim=(0, limit_pat),
        xlabel="Patient mean EU(image) [nats]",
        ylabel="Patient mean EU(image + metadata) [nats]",
        title=f"Per-patient paired EU ({frac_reduced:.1%} reduced)",
    )
    axes[1].grid(alpha=0.25)

    class_names = sorted(frame["diagnostic"].unique())
    x = np.arange(len(class_names) + 1)
    labels_x = ["All", *class_names]
    img_dis = [frame["image_disagreement"].mean()] + [
        frame.loc[frame["diagnostic"] == c, "image_disagreement"].mean()
        for c in class_names
    ]
    mm_dis = [frame["multimodal_disagreement"].mean()] + [
        frame.loc[frame["diagnostic"] == c, "multimodal_disagreement"].mean()
        for c in class_names
    ]
    width = 0.38
    axes[2].bar(x - width / 2, img_dis, width, label="image")
    axes[2].bar(x + width / 2, mm_dis, width, label="multimodal")
    axes[2].set_xticks(x, labels_x, rotation=35, ha="right")
    axes[2].set(
        ylim=(0, 1),
        ylabel="disagreement rate",
        title="Ensemble member prediction disagreement",
    )
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend()

    fig.suptitle(f"{label}: paired EU and member disagreement")
    fig.savefig(path, dpi=150)


def plot_correct_vs_incorrect(
    frame: pd.DataFrame, error_df: pd.DataFrame, label: str, path: Path
) -> None:
    fig = Figure(figsize=(15, 4.5), layout="constrained")
    axes = fig.subplots(1, 3)

    for axis, arm in zip(axes[:2], ("image", "multimodal"), strict=True):
        correct_vals = frame.loc[frame[f"{arm}_correct"], f"{arm}_epistemic"].to_numpy()
        wrong_vals = frame.loc[~frame[f"{arm}_correct"], f"{arm}_epistemic"].to_numpy()
        groups, ticks = [], []
        if len(correct_vals):
            groups.append(correct_vals)
            ticks.append(f"correct\n(n={len(correct_vals)})")
        if len(wrong_vals):
            groups.append(wrong_vals)
            ticks.append(f"incorrect\n(n={len(wrong_vals)})")
        if groups:
            axis.boxplot(groups, tick_labels=ticks, showfliers=False)
        auroc_row = error_df[
            (error_df["arm"] == arm) & (error_df["score"] == "epistemic")
        ]
        auroc = auroc_row["error_auroc"].iloc[0] if not auroc_row.empty else np.nan
        auroc_text = f"{auroc:.3f}" if pd.notna(auroc) else "n/a"
        axis.set(
            ylabel="EU (nats)",
            title=f"{arm}: EU by correctness (error AUROC={auroc_text})",
        )
        axis.grid(alpha=0.25, axis="y")

    groups, ticks = [], []
    for name, mask in (
        ("both right", frame["image_correct"] & frame["multimodal_correct"]),
        ("meta fixed", (~frame["image_correct"]) & frame["multimodal_correct"]),
        ("meta broke", frame["image_correct"] & (~frame["multimodal_correct"])),
        ("both wrong", (~frame["image_correct"]) & (~frame["multimodal_correct"])),
    ):
        vals = frame.loc[mask, "depistemic"].to_numpy()
        if len(vals):
            groups.append(vals)
            ticks.append(f"{name}\n(n={len(vals)})")
    if groups:
        axes[2].boxplot(groups, tick_labels=ticks, showfliers=False)
    axes[2].axhline(0, color="black", linestyle=":")
    axes[2].set(ylabel="dEU (nats)", title="dEU by prediction outcome")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle(f"{label}: uncertainty vs prediction correctness")
    fig.savefig(path, dpi=150)


def plot_risk_coverage(
    curves_df: pd.DataFrame, aurc_df: pd.DataFrame, label: str, path: Path
) -> None:
    fig = Figure(figsize=(13, 4.8), layout="constrained")
    axes = fig.subplots(1, 2)
    for axis, arm in zip(axes, ("image", "multimodal"), strict=True):
        arm_curves = curves_df[curves_df["arm"] == arm]
        arm_aurc = aurc_df[aurc_df["arm"] == arm].set_index("score")["aurc"].to_dict()
        for score_name, group in arm_curves.groupby("score"):
            aurc = arm_aurc.get(score_name, np.nan)
            axis.plot(
                group["coverage"],
                group["risk"],
                marker="o",
                markersize=3,
                label=f"{score_name} (AURC={aurc:.3f})",
            )
        axis.set(
            xlim=(0.1, 1.02),
            ylim=(0, 1),
            xlabel="coverage (fraction kept)",
            ylabel="error rate (risk)",
            title=f"{arm}: selective prediction",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"{label}: risk-coverage curves")
    fig.savefig(path, dpi=150)


def plot_classification(
    metrics: dict[str, dict], classes: list[str], label: str, path: Path
) -> None:
    fig = Figure(figsize=(16, 9.2), layout="constrained")
    axes = fig.subplots(2, 2)

    for axis, arm in zip(axes[0], ("image", "multimodal"), strict=True):
        confusion = np.asarray(metrics[arm]["confusion_matrix"], dtype=float)
        support = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion, support, out=np.zeros_like(confusion), where=support > 0
        )
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ticks = range(len(classes))
        axis.set_xticks(ticks, classes, rotation=45)
        axis.set_yticks(
            ticks, [f"{c} (n={int(n)})" for c, n in zip(classes, support[:, 0], strict=True)]
        )
        axis.set(title=f"{arm} confusion matrix", xlabel="predicted", ylabel="true")
        for i in ticks:
            for j in ticks:
                colour = "white" if normalized[i, j] > 0.5 else "black"
                axis.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color=colour)
        fig.colorbar(image, ax=axis, label="row fraction")

    names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mel_recall",
        "macro_ovr_auroc",
    ]
    present_names = [n for n in names if metrics["image"].get(n) is not None]
    x = np.arange(len(present_names))
    width = 0.38
    axes[1, 0].bar(
        x - width / 2, [metrics["image"][n] for n in present_names], width, label="image"
    )
    axes[1, 0].bar(
        x + width / 2,
        [metrics["multimodal"][n] for n in present_names],
        width,
        label="multimodal",
    )
    axes[1, 0].set_xticks(
        x, [n.replace("_", " ") for n in present_names], rotation=30, ha="right"
    )
    axes[1, 0].set(ylim=(0, 1), title="Overall classification metrics")
    axes[1, 0].grid(alpha=0.25, axis="y")
    axes[1, 0].legend()

    cx = np.arange(len(classes))
    w = 0.2
    img_rep = metrics["image"]["classification_report"]
    mm_rep = metrics["multimodal"]["classification_report"]
    axes[1, 1].bar(
        cx - 1.5 * w, [img_rep[c]["recall"] for c in classes], w, label="image recall"
    )
    axes[1, 1].bar(
        cx - 0.5 * w, [mm_rep[c]["recall"] for c in classes], w, label="multimodal recall"
    )
    axes[1, 1].bar(
        cx + 0.5 * w, [img_rep[c]["f1-score"] for c in classes], w, label="image F1"
    )
    axes[1, 1].bar(
        cx + 1.5 * w, [mm_rep[c]["f1-score"] for c in classes], w, label="multimodal F1"
    )
    axes[1, 1].set_xticks(cx, classes)
    axes[1, 1].set(ylim=(0, 1), title="Per-class recall and F1")
    axes[1, 1].grid(alpha=0.25, axis="y")
    axes[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle(f"{label}: classification performance")
    fig.savefig(path, dpi=150)


def _reliability_points(
    frame: pd.DataFrame, arm: str, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    probability_columns = [column for column in frame if column.startswith(f"{arm}_p_")]
    probs = frame[probability_columns].to_numpy()
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == frame["label"].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    mean_confidence, accuracy = [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            include |= confidence == 0
        if include.any():
            mean_confidence.append(confidence[include].mean())
            accuracy.append(correct[include].mean())
    return np.asarray(mean_confidence), np.asarray(accuracy)


def plot_calibration(
    frame: pd.DataFrame, metrics: dict[str, dict], label: str, path: Path
) -> None:
    fig = Figure(figsize=(6.8, 5.4), layout="constrained")
    axis = fig.subplots()
    axis.plot([0, 1], [0, 1], ":", color="black", label="perfect")
    for arm in ("image", "multimodal"):
        confidence, accuracy = _reliability_points(frame, arm)
        m = metrics[arm]
        legend = (
            f"{arm} (ECE={m['ece']:.3f}, NLL={m['negative_log_likelihood']:.3f}, "
            f"Brier={m['multiclass_brier']:.3f})"
        )
        axis.plot(confidence, accuracy, marker="o", label=legend)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="mean confidence",
        ylabel="empirical accuracy",
        title=f"{label}: calibration reliability diagram",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)


def plot_training_curves(curves: pd.DataFrame, label: str, path: Path) -> None:
    panels = [
        ("loss", "Loss", None),
        ("accuracy", "Accuracy", (0, 1)),
        ("balanced_accuracy", "Balanced accuracy", (0, 1)),
        ("macro_f1", "Macro-F1", (0, 1)),
    ]
    fig = Figure(figsize=(16, 4.4), layout="constrained")
    axes = fig.subplots(1, len(panels))
    arm_colors = {"image": "C0", "multimodal": "C1"}
    for axis, (metric, title, ylim) in zip(axes, panels, strict=True):
        for arm in ("image", "multimodal"):
            sub = curves[curves["arm"] == arm].sort_values("epoch")
            if sub.empty:
                continue
            color = arm_colors[arm]
            for split, style in (("train", "--"), ("val", "-")):
                mean_col = f"{split}_{metric}_mean"
                std_col = f"{split}_{metric}_std"
                if mean_col not in sub or sub[mean_col].isna().all():
                    continue
                epochs = sub["epoch"].to_numpy()
                mean = sub[mean_col].to_numpy()
                std = sub[std_col].fillna(0.0).to_numpy()
                axis.plot(
                    epochs, mean, style, color=color, marker="o", markersize=3,
                    label=f"{arm} {split}",
                )
                axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12)
        axis.set(xlabel="epoch", title=title, ylim=ylim)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(f"{label}: aggregated training curves (mean +/- 1 std across folds and members)")
    fig.savefig(path, dpi=150)


def plot_fold_comparison(fold_df: pd.DataFrame, label: str, path: Path) -> None:
    ordered = fold_df.sort_values("fold")
    x = np.arange(len(ordered))
    ticks = [f"fold {int(f)}" for f in ordered["fold"]]
    width = 0.36

    fig = Figure(figsize=(16, 4.6), layout="constrained")
    axes = fig.subplots(1, 3)

    axes[0].bar(x - width / 2, ordered["mean_image_epistemic"], width, label="image")
    axes[0].bar(x + width / 2, ordered["mean_multimodal_epistemic"], width, label="multimodal")
    axes[0].set_xticks(x, ticks)
    axes[0].set(ylabel="patient mean EU (nats)", title="EU per fold")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    axes[1].errorbar(
        x,
        ordered["mean_deu"],
        yerr=[
            np.maximum(ordered["mean_deu"] - ordered["deu_ci_low"], 0),
            np.maximum(ordered["deu_ci_high"] - ordered["mean_deu"], 0),
        ],
        fmt="o",
        capsize=4,
    )
    axes[1].axhline(0, color="black", linestyle=":")
    axes[1].set_xticks(x, ticks)
    axes[1].set(ylabel="dEU (nats)", title="Patient mean dEU per fold (95% CI)")
    axes[1].grid(alpha=0.25, axis="y")

    axes[2].bar(x - width / 2, ordered["image_balanced_accuracy"], width, label="image")
    axes[2].bar(
        x + width / 2, ordered["multimodal_balanced_accuracy"], width, label="multimodal"
    )
    axes[2].set_xticks(x, ticks)
    axes[2].set(ylim=(0, 1), ylabel="balanced accuracy", title="Balanced accuracy per fold")
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend()

    fig.suptitle(f"{label}: fold-by-fold stability")
    fig.savefig(path, dpi=150)


def plot_ensemble_size_stability(stability_df: pd.DataFrame, label: str, path: Path) -> None:
    ordered = stability_df.sort_values("n_members")
    m = ordered["n_members"].to_numpy()
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    axes = fig.subplots(1, 2)

    for arm in ("image", "multimodal"):
        mean = ordered[f"{arm}_eu_mean"].to_numpy()
        std = ordered[f"{arm}_eu_std"].to_numpy()
        axes[0].errorbar(m, mean, yerr=std, marker="o", capsize=4, label=arm)
    axes[0].set_xticks(m)
    axes[0].set(
        xlabel="ensemble size M",
        ylabel="patient mean EU (nats)",
        title="EU stability vs ensemble size",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].errorbar(
        m,
        ordered["mean_deu_mean"],
        yerr=ordered["mean_deu_std"],
        marker="o",
        capsize=4,
        color="C2",
    )
    axes[1].axhline(0, color="black", linestyle=":")
    axes[1].set_xticks(m)
    axes[1].set(
        xlabel="ensemble size M",
        ylabel="patient mean dEU (nats)",
        title="dEU stability across seed subsets",
    )
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{label}: post-hoc ensemble size stability")
    fig.savefig(path, dpi=150)


def plot_missing_metadata_stratification(
    strata_df: pd.DataFrame, label: str, path: Path
) -> None:
    fig = Figure(figsize=(13, 4.6), layout="constrained")
    axes = fig.subplots(1, 2)
    x = np.arange(len(strata_df))
    ticks = [
        f"{row['stratum']}\n(n={int(row['n_images'])})"
        for _, row in strata_df.iterrows()
    ]
    axes[0].errorbar(
        x,
        strata_df["mean_deu"],
        yerr=[
            np.maximum(strata_df["mean_deu"] - strata_df["deu_ci_low"], 0),
            np.maximum(strata_df["deu_ci_high"] - strata_df["mean_deu"], 0),
        ],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="black", linestyle=":")
    axes[0].set_xticks(x, ticks, rotation=20, ha="right")
    axes[0].set(ylabel="mean dEU (nats)", title="dEU by metadata completeness / unit")
    axes[0].grid(alpha=0.25, axis="y")

    width = 0.36
    axes[1].bar(
        x - width / 2, strata_df["image_balanced_accuracy"], width, label="image"
    )
    axes[1].bar(
        x + width / 2,
        strata_df["multimodal_balanced_accuracy"],
        width,
        label="multimodal",
    )
    axes[1].set_xticks(x, ticks, rotation=20, ha="right")
    axes[1].set(ylim=(0, 1), ylabel="balanced accuracy", title="Balanced accuracy by stratum")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()

    fig.suptitle(f"{label}: clinical metadata completeness and aggregation unit")
    fig.savefig(path, dpi=150)


def plot_condition_comparison(comparison: pd.DataFrame, path: Path) -> None:
    fig = Figure(figsize=(15, 5), layout="constrained")
    axes = fig.subplots(1, 3)
    x = np.arange(len(comparison))
    width = 0.38

    axes[0].errorbar(
        x,
        comparison["mean_deu"],
        yerr=[
            np.maximum(comparison["mean_deu"] - comparison["deu_ci_low"], 0),
            np.maximum(comparison["deu_ci_high"] - comparison["mean_deu"], 0),
        ],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="black", linestyle=":")
    axes[0].set(title="Mean patient dEU (95% CI)", ylabel="nats")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(
        x - width / 2, comparison["image_balanced_accuracy"], width, label="image"
    )
    axes[1].bar(
        x + width / 2,
        comparison["multimodal_balanced_accuracy"],
        width,
        label="multimodal",
    )
    axes[1].set(title="Balanced accuracy", ylim=(0, 1))
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()

    axes[2].bar(x - width / 2, comparison["image_macro_f1"], width, label="image")
    axes[2].bar(
        x + width / 2, comparison["multimodal_macro_f1"], width, label="multimodal"
    )
    axes[2].set(title="Macro-F1", ylim=(0, 1))
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend()

    for axis in axes:
        axis.set_xticks(x, comparison["condition"], rotation=30, ha="right")
    fig.savefig(path, dpi=150)


def plot_eu_measures_comparison(comparison: pd.DataFrame, path: Path) -> None:
    panels = [
        ("epistemic", "Entropy EU (MI, nats)"),
        ("variance_epistemic", "Variance EU"),
        ("normalized_epistemic", "Normalized EU / TU"),
        ("total", "Total uncertainty (nats)"),
        ("aleatoric", "Aleatoric uncertainty (nats)"),
    ]
    fig = Figure(figsize=(16, 8.2), layout="constrained")
    axes = fig.subplots(2, 3).ravel()
    x = np.arange(len(comparison))
    width = 0.36
    for axis, (suffix, title) in zip(axes[: len(panels)], panels, strict=True):
        axis.bar(x - width / 2, comparison[f"mean_image_{suffix}"], width, label="image")
        axis.bar(
            x + width / 2, comparison[f"mean_multimodal_{suffix}"], width, label="multimodal"
        )
        axis.set_xticks(x, comparison["condition"], rotation=30, ha="right")
        axis.set(title=title)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)

    axes[5].bar(x - width / 2, comparison["mean_deu"], width, label="MI dEU")
    axes[5].bar(
        x + width / 2, comparison["mean_dvariance_epistemic"], width, label="Variance dEU"
    )
    axes[5].axhline(0, color="black", linestyle=":")
    axes[5].set_xticks(x, comparison["condition"], rotation=30, ha="right")
    axes[5].set(title="dEU across uncertainty measures")
    axes[5].grid(alpha=0.25, axis="y")
    axes[5].legend(fontsize=8)
    fig.suptitle("Uncertainty decomposition and EU measure comparison across setups")
    fig.savefig(path, dpi=150)


def plot_cross_setup_fold_eu(all_folds: pd.DataFrame, path: Path) -> None:
    conditions = list(dict.fromkeys(all_folds["condition"]))
    fig = Figure(figsize=(15, 4.8), layout="constrained")
    axes = fig.subplots(1, 3)

    for condition in conditions:
        sub = all_folds[all_folds["condition"] == condition].sort_values("fold")
        axes[0].plot(
            sub["fold"], sub["mean_image_epistemic"], marker="o", label=condition
        )
        axes[1].plot(
            sub["fold"], sub["mean_multimodal_epistemic"], marker="o", label=condition
        )
        axes[2].errorbar(
            sub["fold"],
            sub["mean_deu"],
            yerr=[
                np.maximum(sub["mean_deu"] - sub["deu_ci_low"], 0),
                np.maximum(sub["deu_ci_high"] - sub["mean_deu"], 0),
            ],
            marker="o",
            capsize=3,
            label=condition,
        )
    axes[0].set(xlabel="fold", ylabel="patient mean EU (nats)", title="Image-only EU by fold")
    axes[1].set(xlabel="fold", ylabel="patient mean EU (nats)", title="Multimodal EU by fold")
    axes[2].axhline(0, color="black", linestyle=":")
    axes[2].set(xlabel="fold", ylabel="patient mean dEU (nats)", title="Paired dEU by fold")
    for axis in axes:
        axis.set_xticks(sorted(all_folds["fold"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Fold-by-fold uncertainty comparison across experiment setups")
    fig.savefig(path, dpi=150)


def plot_cross_setup_learning_curves(all_curves: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("val_loss", "Validation loss", None),
        ("val_accuracy", "Validation accuracy", (0, 1)),
        ("val_balanced_accuracy", "Validation balanced accuracy", (0, 1)),
        ("val_macro_f1", "Validation macro-F1", (0, 1)),
    ]
    fig = Figure(figsize=(16, 4.6), layout="constrained")
    axes = fig.subplots(1, len(metrics))
    conditions = list(dict.fromkeys(all_curves["condition"]))
    styles = {"image": "--", "multimodal": "-"}
    for axis, (metric, title, ylim) in zip(axes, metrics, strict=True):
        mean_col = f"{metric}_mean"
        for idx, condition in enumerate(conditions):
            color = f"C{idx % 10}"
            for arm in ("image", "multimodal"):
                sub = all_curves[
                    (all_curves["condition"] == condition) & (all_curves["arm"] == arm)
                ].sort_values("epoch")
                if sub.empty or mean_col not in sub or sub[mean_col].isna().all():
                    continue
                axis.plot(
                    sub["epoch"],
                    sub[mean_col],
                    styles[arm],
                    color=color,
                    marker="o",
                    markersize=3,
                    label=f"{condition} ({arm})",
                )
        axis.set(xlabel="epoch", title=title, ylim=ylim)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(
        "Validation learning curves across experiment setups (dashed=image, solid=multimodal)"
    )
    fig.savefig(path, dpi=150)


def plot_train_fraction(comparison: pd.DataFrame, path: Path) -> None:
    fig = Figure(figsize=(16, 8.6), layout="constrained")
    axes = fig.subplots(2, 3)

    for architecture, group in comparison.groupby("architecture"):
        group = group.sort_values("train_fraction")
        frac = group["train_fraction"]
        for arm in ("image", "multimodal"):
            tag = f"{architecture} {arm}"
            axes[0, 0].plot(frac, group[f"mean_{arm}_epistemic"], marker="o", label=tag)
            axes[0, 1].plot(frac, group[f"mean_{arm}_variance_epistemic"], marker="o", label=tag)
            axes[0, 2].plot(frac, group[f"mean_{arm}_aleatoric"], marker="o", label=tag)
            axes[1, 1].plot(frac, group[f"{arm}_balanced_accuracy"], marker="o", label=tag)
            axes[1, 2].plot(frac, group[f"{arm}_macro_f1"], marker="o", label=tag)
        axes[1, 0].errorbar(
            frac,
            group["mean_deu"],
            yerr=[
                np.maximum(group["mean_deu"] - group["deu_ci_low"], 0),
                np.maximum(group["deu_ci_high"] - group["mean_deu"], 0),
            ],
            marker="o",
            capsize=4,
            label=str(architecture),
        )

    axes[0, 0].set(title="MI Epistemic Uncertainty", ylabel="nats")
    axes[0, 1].set(title="Variance Epistemic Uncertainty", ylabel="sum variance")
    axes[0, 2].set(title="Aleatoric Uncertainty", ylabel="nats")
    axes[1, 0].axhline(0, color="black", linestyle=":")
    axes[1, 0].set(title="Paired dEU (95% CI)", ylabel="nats")
    axes[1, 1].set(title="Balanced Accuracy", ylabel="score", ylim=(0, 1))
    axes[1, 2].set(title="Macro-F1", ylabel="score", ylim=(0, 1))

    for axis in axes.ravel():
        axis.set_xlabel("training-patient fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Positive control: epistemic uncertainty and classification vs training-set size")
    fig.savefig(path, dpi=150)

"""Build the standardized result-analysis directory from saved ensemble predictions."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import CLASSES
from src.analysis.effects import (
    class_scores,
    patient_cluster_mean_ci,
    performance_row,
    reduction_stats,
)
from src.analysis.figures import (
    plot_confusion_matrices,
    plot_deu_candles,
    plot_deu_distribution,
    plot_effect_estimates,
    plot_paired_eu_scatter,
    plot_setting_uncertainty_change,
    plot_total_vs_aleatoric,
    plot_validation_curves,
)
from src.metrics import detailed_metrics
from src.uq.deep_ensemble import uncertainty

REPO = Path(__file__).resolve().parents[2]


@dataclass
class FoldPrediction:
    fold: int
    directory: Path
    probabilities: np.ndarray
    labels: np.ndarray
    img_ids: np.ndarray
    patient_ids: np.ndarray
    seeds: list[int]
    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray

    @property
    def mean_probabilities(self) -> np.ndarray:
        return self.probabilities.mean(axis=0)

    @property
    def predicted(self) -> np.ndarray:
        return self.mean_probabilities.argmax(axis=1)


@dataclass
class Condition:
    run_group: str
    label: str
    folds: list[FoldPrediction]


@dataclass
class Comparison:
    comparison_id: str
    name: str
    reference: Condition
    candidate: Condition


def _as_text(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in {"", "None", "null"}:
        return None
    return text


def _split_badge(split: str) -> str:
    if split == "val":
        return "Validation split"
    if split == "test":
        return "Held-out test split"
    raise ValueError("eval_split must be 'val' or 'test'")


def _infer_label(run_group: str, override: str | None) -> str:
    if override:
        return override
    if "image_only" in run_group:
        return "Image-only"
    if "multimodal" in run_group:
        return "Image + Metadata"
    return run_group


def _load_fold(directory: Path, fold: int, split: str) -> FoldPrediction:
    path = directory / f"{split}_member_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    with np.load(path, allow_pickle=False) as saved:
        classes = [str(item) for item in saved["classes"]]
        if classes != list(CLASSES):
            raise ValueError(f"{path} class order {classes} != {list(CLASSES)}")
        probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
        total, aleatoric, epistemic = uncertainty(probabilities)
        return FoldPrediction(
            fold=fold,
            directory=directory,
            probabilities=probabilities,
            labels=np.asarray(saved["labels"]),
            img_ids=np.asarray(saved["img_ids"]).astype(str),
            patient_ids=np.asarray(saved["patient_ids"]).astype(str),
            seeds=[int(seed) for seed in saved["seeds"]],
            total=total,
            aleatoric=aleatoric,
            epistemic=epistemic,
        )


def load_condition(
    runs_root: Path, run_group: str, label: str, n_folds: int, split: str
) -> Condition:
    folds = []
    for fold in range(n_folds):
        directory = runs_root / run_group / f"fold_{fold}"
        folds.append(_load_fold(directory, fold, split))
    sizes = {len(fold.seeds) for fold in folds}
    if len(sizes) != 1:
        raise ValueError(f"{run_group} has different ensemble sizes across folds: {sorted(sizes)}")
    return Condition(run_group=run_group, label=label, folds=folds)


def _concat(condition: Condition) -> FoldPrediction:
    """Stack folds into one prediction object. The directory is the first fold."""
    return FoldPrediction(
        fold=-1,
        directory=condition.folds[0].directory,
        probabilities=np.concatenate([fold.probabilities for fold in condition.folds], axis=1),
        labels=np.concatenate([fold.labels for fold in condition.folds]),
        img_ids=np.concatenate([fold.img_ids for fold in condition.folds]),
        patient_ids=np.concatenate([fold.patient_ids for fold in condition.folds]),
        seeds=condition.folds[0].seeds,
        total=np.concatenate([fold.total for fold in condition.folds]),
        aleatoric=np.concatenate([fold.aleatoric for fold in condition.folds]),
        epistemic=np.concatenate([fold.epistemic for fold in condition.folds]),
    )


def _pair_fold(
    comparison: Comparison, reference: FoldPrediction, candidate: FoldPrediction
) -> pd.DataFrame:
    if reference.seeds != candidate.seeds:
        raise ValueError(
            f"{comparison.name} fold {reference.fold} ensemble seeds differ: "
            f"{reference.seeds} vs {candidate.seeds}"
        )
    if len(reference.img_ids) != len(set(reference.img_ids)) or len(candidate.img_ids) != len(
        set(candidate.img_ids)
    ):
        raise ValueError(f"{comparison.name} fold {reference.fold} has duplicate image IDs")
    left = pd.DataFrame(
        {
            "img_id": reference.img_ids,
            "patient_id": reference.patient_ids,
            "label": reference.labels,
            "reference_predicted": reference.predicted,
            "reference_correct": reference.predicted == reference.labels,
            "reference_total": reference.total,
            "reference_aleatoric": reference.aleatoric,
            "reference_epistemic": reference.epistemic,
        }
    )
    right = pd.DataFrame(
        {
            "img_id": candidate.img_ids,
            "patient_id": candidate.patient_ids,
            "label": candidate.labels,
            "candidate_predicted": candidate.predicted,
            "candidate_correct": candidate.predicted == candidate.labels,
            "candidate_total": candidate.total,
            "candidate_aleatoric": candidate.aleatoric,
            "candidate_epistemic": candidate.epistemic,
        }
    )
    merged = left.merge(right, on="img_id", how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{comparison.name} fold {reference.fold} image IDs do not match")
    if (
        not merged["patient_id_x"].eq(merged["patient_id_y"]).all()
        or not merged["label_x"].eq(merged["label_y"]).all()
    ):
        raise ValueError(
            f"{comparison.name} fold {reference.fold} patient IDs or labels do not match"
        )
    merged = merged.drop(columns=["_merge", "patient_id_y", "label_y"]).rename(
        columns={"patient_id_x": "patient_id", "label_x": "label"}
    )
    merged.insert(0, "comparison", comparison.name)
    merged.insert(1, "comparison_id", comparison.comparison_id)
    merged.insert(2, "fold", reference.fold)
    merged["diagnosis"] = [CLASSES[int(label)] for label in merged["label"]]
    merged["total_reduction"] = merged["reference_total"] - merged["candidate_total"]
    merged["aleatoric_reduction"] = merged["reference_aleatoric"] - merged["candidate_aleatoric"]
    merged["epistemic_reduction"] = merged["reference_epistemic"] - merged["candidate_epistemic"]
    return merged


def pair_comparison(comparison: Comparison) -> pd.DataFrame:
    by_fold = {fold.fold: fold for fold in comparison.candidate.folds}
    frames = []
    for reference in comparison.reference.folds:
        if reference.fold not in by_fold:
            raise ValueError(f"{comparison.name} is missing candidate fold {reference.fold}")
        frames.append(_pair_fold(comparison, reference, by_fold[reference.fold]))
    return pd.concat(frames, ignore_index=True)


def _class_count_row(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts()
    return {f"n_{name}": int(counts.get(name, 0)) for name in CLASSES}


def cohort_table(comparisons: list[Comparison], paired: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for comparison in comparisons:
        frame = paired[comparison.comparison_id]
        pieces = [
            ("pooled", frame),
            *[(str(fold), part) for fold, part in frame.groupby("fold", sort=True)],
        ]
        for fold_name, part in pieces:
            rows.append(
                {
                    "comparison": comparison.name,
                    "fold": fold_name,
                    "reference_run": comparison.reference.run_group,
                    "candidate_run": comparison.candidate.run_group,
                    "reference_label": comparison.reference.label,
                    "candidate_label": comparison.candidate.label,
                    "ensemble_size": len(comparison.reference.folds[0].seeds),
                    "paired_images": len(part),
                    "paired_patients": int(part["patient_id"].nunique()),
                    **_class_count_row(part["diagnosis"]),
                }
            )
    return pd.DataFrame(rows)


def performance_table(conditions: list[Condition]) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        pieces: list[tuple[str, FoldPrediction]] = [("pooled", _concat(condition))]
        pieces.extend((str(fold.fold), fold) for fold in condition.folds)
        for fold_name, fold in pieces:
            rows.append(
                {
                    "condition": condition.label,
                    "run_group": condition.run_group,
                    "fold": fold_name,
                    **performance_row(
                        fold.labels,
                        fold.mean_probabilities,
                        fold.total,
                        fold.aleatoric,
                        fold.epistemic,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pooled_metrics(condition: Condition) -> dict:
    folded = _concat(condition)
    return performance_row(
        folded.labels, folded.mean_probabilities, folded.total, folded.aleatoric, folded.epistemic
    )


def paired_effects_table(
    comparisons: list[Comparison], paired: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for comparison in comparisons:
        frame = paired[comparison.comparison_id]
        reference = _pooled_metrics(comparison.reference)
        candidate = _pooled_metrics(comparison.candidate)
        row = {
            "comparison": comparison.name,
            "paired_images": len(frame),
            "paired_patients": int(frame["patient_id"].nunique()),
            "delta_balanced_accuracy_pp": (
                candidate["balanced_accuracy_pct"] - reference["balanced_accuracy_pct"]
            ),
            "delta_macro_f1_pp": candidate["macro_f1_pct"] - reference["macro_f1_pct"],
        }
        mapping = {
            "total_uncertainty": ("reference_total", "candidate_total"),
            "aleatoric_uncertainty": ("reference_aleatoric", "candidate_aleatoric"),
            "epistemic_uncertainty": ("reference_epistemic", "candidate_epistemic"),
        }
        for quantity, (reference_column, candidate_column) in mapping.items():
            stats = reduction_stats(
                frame[reference_column].to_numpy(), frame[candidate_column].to_numpy()
            )
            row[f"mean_{quantity}_reduction_nats"] = stats["mean_nats"]
            row[f"median_{quantity}_reduction_nats"] = stats["median_nats"]
            row[f"relative_{quantity}_reduction_pct"] = stats["relative_pct"]
            row[f"share_{quantity}_reduced_pct"] = stats["share_reduced_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


_SETTING_CHANGE_IDS = (
    "image_only_change",
    "multimodal_change",
    "baseline_modality",
    "candidate_modality",
)
_SETTING_DELTAS = (
    ("image_only_change", "aleatoric_reduction", "dAU_image_only", "dAU (image-only)"),
    ("image_only_change", "epistemic_reduction", "dEU_image_only", "dEU (image-only)"),
    ("image_only_change", "total_reduction", "dTU_image_only", "dTU (image-only)"),
    ("multimodal_change", "aleatoric_reduction", "dAU_multimodal", "dAU (image + metadata)"),
    ("multimodal_change", "epistemic_reduction", "dEU_multimodal", "dEU (image + metadata)"),
    ("multimodal_change", "total_reduction", "dTU_multimodal", "dTU (image + metadata)"),
)


def setting_uncertainty_change(
    paired: dict[str, pd.DataFrame], *, samples: int, seed: int
) -> pd.DataFrame | None:
    """Uncertainty change from the baseline setting to the candidate setting.

    dTU, dAU, and dEU are baseline minus candidate. A positive value means the
    candidate setting is more certain. d(dEU) is the candidate metadata EU
    reduction minus the baseline metadata EU reduction. A positive value means
    metadata reduces epistemic uncertainty by more under the candidate setting.
    """
    if any(key not in paired for key in _SETTING_CHANGE_IDS):
        return None
    rows = []
    for offset, (comparison_id, column, quantity, label) in enumerate(_SETTING_DELTAS):
        frame = paired[comparison_id]
        quantity_name = column.removesuffix("_reduction")
        stats = reduction_stats(
            frame[f"reference_{quantity_name}"].to_numpy(),
            frame[f"candidate_{quantity_name}"].to_numpy(),
        )
        _, low, high = patient_cluster_mean_ci(
            frame["patient_id"].to_numpy(),
            frame[column].to_numpy(),
            samples=samples,
            seed=seed + offset,
        )
        rows.append(_setting_change_row(quantity, label, frame, stats, low, high))

    baseline = paired["baseline_modality"][
        ["fold", "img_id", "patient_id", "epistemic_reduction"]
    ].rename(columns={"epistemic_reduction": "baseline_deu"})
    candidate = paired["candidate_modality"][["fold", "img_id", "epistemic_reduction"]].rename(
        columns={"epistemic_reduction": "candidate_deu"}
    )
    joined = baseline.merge(candidate, on=["fold", "img_id"], how="outer", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("baseline and candidate modality effects do not share image IDs")
    delta = joined["candidate_deu"].to_numpy() - joined["baseline_deu"].to_numpy()
    denominator = float(np.mean(joined["baseline_deu"]))
    relative: float | str = (
        "unavailable" if denominator == 0 else 100.0 * float(delta.mean()) / denominator
    )
    _, low, high = patient_cluster_mean_ci(
        joined["patient_id"].to_numpy(),
        delta,
        samples=samples,
        seed=seed + len(_SETTING_DELTAS),
    )
    rows.append(
        _setting_change_row(
            "d_dEU",
            "d(dEU)",
            joined,
            {
                "mean_nats": float(delta.mean()),
                "median_nats": float(np.median(delta)),
                "relative_pct": relative,
                "share_reduced_pct": 100.0 * float(np.mean(delta > 0)),
            },
            low,
            high,
        )
    )
    return pd.DataFrame(rows)


def _setting_change_row(
    quantity: str,
    label: str,
    frame: pd.DataFrame,
    stats: dict,
    low: float,
    high: float,
) -> dict:
    return {
        "quantity": quantity,
        "label": label,
        "paired_images": len(frame),
        "paired_patients": int(frame["patient_id"].nunique()),
        "mean_nats": stats["mean_nats"],
        "median_nats": stats["median_nats"],
        "relative_pct": stats["relative_pct"],
        "share_positive_pct": stats["share_reduced_pct"],
        "ci_low_nats": low,
        "ci_high_nats": high,
    }


def class_breakdown_table(
    conditions: list[Condition], comparisons: list[Comparison], paired: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        folded = _concat(condition)
        scores = class_scores(folded.labels, folded.predicted, CLASSES)
        for record in scores.to_dict(orient="records"):
            rows.append(
                {
                    "row_type": "condition",
                    "name": condition.label,
                    **record,
                    "mean_epistemic_reduction_nats": np.nan,
                    "paired_images": np.nan,
                }
            )
    for comparison in comparisons:
        frame = paired[comparison.comparison_id]
        for diagnosis in CLASSES:
            subset = frame[frame["diagnosis"] == diagnosis]
            rows.append(
                {
                    "row_type": "comparison",
                    "name": comparison.name,
                    "diagnosis": diagnosis,
                    "support": len(subset),
                    "precision_pct": np.nan,
                    "recall_pct": np.nan,
                    "f1_pct": np.nan,
                    "mean_epistemic_reduction_nats": float(subset["epistemic_reduction"].mean())
                    if len(subset)
                    else np.nan,
                    "paired_images": len(subset),
                }
            )
    return pd.DataFrame(rows)


def load_validation_history(conditions: list[Condition]) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        for fold in condition.folds:
            for seed in fold.seeds:
                history_path = fold.directory / "members" / f"seed_{seed}" / "history.csv"
                if not history_path.exists():
                    raise FileNotFoundError(f"missing {history_path}")
                history = pd.read_csv(history_path)
                required = {"epoch", "val_loss", "val_accuracy"}
                missing = required - set(history.columns)
                if missing:
                    raise ValueError(f"{history_path} is missing {sorted(missing)}")
                for record in history.to_dict(orient="records"):
                    rows.append(
                        {
                            "setup": condition.label,
                            "fold": fold.fold,
                            "member_seed": seed,
                            "epoch": int(record["epoch"]),
                            "val_loss": float(record["val_loss"]),
                            "val_accuracy_pct": 100.0 * float(record["val_accuracy"]),
                        }
                    )
    if not rows:
        raise ValueError("no validation histories were found")
    return (
        pd.DataFrame(rows)
        .sort_values(["setup", "fold", "member_seed", "epoch"])
        .reset_index(drop=True)
    )


def _confusion(condition: Condition) -> np.ndarray:
    folded = _concat(condition)
    return np.array(detailed_metrics(folded.labels, folded.predicted, CLASSES)["confusion_matrix"])


def _effect_rows(
    comparisons: list[Comparison], paired: dict[str, pd.DataFrame], samples: int, seed: int
) -> list[dict]:
    rows = []
    for index, comparison in enumerate(comparisons):
        frame = paired[comparison.comparison_id]
        mean, low, high = patient_cluster_mean_ci(
            frame["patient_id"].to_numpy(),
            frame["epistemic_reduction"].to_numpy(),
            samples=samples,
            seed=seed + index,
        )
        rows.append({"label": comparison.name, "mean": mean, "low": low, "high": high})
        for class_index, diagnosis in enumerate(CLASSES):
            subset = frame[frame["diagnosis"] == diagnosis]
            if subset.empty:
                continue
            mean, low, high = patient_cluster_mean_ci(
                subset["patient_id"].to_numpy(),
                subset["epistemic_reduction"].to_numpy(),
                samples=samples,
                seed=seed + 100 + index * 10 + class_index,
            )
            rows.append(
                {
                    "label": f"{comparison.name} · {diagnosis}",
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    return rows


def _settings(directory: Path) -> dict:
    path = directory / "config.yaml"
    if not path.exists():
        return {}
    cfg = OmegaConf.load(path)
    scheduler = cfg.train.get("scheduler")
    scheduler_name = (
        None if scheduler is None else str(scheduler.get("_target_", "none")).rsplit(".", 1)[-1]
    )
    augmentation = cfg.data.get("augmentation")
    return {
        "learning_rate": float(cfg.train.optimizer.lr),
        "scheduler": scheduler_name,
        "weighted_loss": bool(cfg.train.weighted_loss),
        "class_weight_power": float(cfg.train.class_weight_power),
        "label_smoothing": float(cfg.train.loss.get("label_smoothing", 0.0)),
        "max_epochs": int(cfg.train.epochs),
        "monitor": str(cfg.train.monitor),
        "early_stopping_patience": cfg.train.get("early_stopping_patience"),
        "early_stopping_min_delta": float(cfg.train.early_stopping_min_delta),
        "freeze_backbone": bool(cfg.model.image_encoder.freeze),
        "augmentation": None
        if augmentation is None
        else {
            "horizontal_flip": float(augmentation.horizontal_flip),
            "vertical_flip": float(augmentation.vertical_flip),
            "rotation_degrees": float(augmentation.rotation_degrees),
            "crop_scale": None
            if augmentation.crop_scale is None
            else [float(v) for v in augmentation.crop_scale],
            "brightness": float(augmentation.brightness),
            "contrast": float(augmentation.contrast),
            "saturation": float(augmentation.saturation),
            "hue": float(augmentation.hue),
        },
    }


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cell(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}" if np.isfinite(value) else ""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    return _markdown_rows(list(selected.columns), selected.itertuples(index=False, name=None))


def _markdown_rows(headers: list[str], rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _paired_effects_markdown(effects: pd.DataFrame) -> str:
    """One narrow block per comparison. The wide CSV keeps the full column names."""
    quantities = (
        ("total_uncertainty", "Total"),
        ("aleatoric_uncertainty", "Aleatoric"),
        ("epistemic_uncertainty", "Epistemic"),
    )
    blocks = []
    for record in effects.to_dict(orient="records"):
        uncertainty_rows = [
            [
                label,
                record[f"mean_{key}_reduction_nats"],
                record[f"median_{key}_reduction_nats"],
                record[f"relative_{key}_reduction_pct"],
                record[f"share_{key}_reduced_pct"],
            ]
            for key, label in quantities
        ]
        blocks.append(
            "\n".join(
                [
                    f"### {record['comparison']}",
                    "",
                    _markdown_rows(
                        ["Images", "Patients", "Δ bal. acc. (pp)", "Δ macro-F1 (pp)"],
                        [
                            [
                                record["paired_images"],
                                record["paired_patients"],
                                record["delta_balanced_accuracy_pp"],
                                record["delta_macro_f1_pp"],
                            ]
                        ],
                    ),
                    "",
                    _markdown_rows(
                        [
                            "Uncertainty",
                            "Mean (nats)",
                            "Median (nats)",
                            "Relative (%)",
                            "Share reduced (%)",
                        ],
                        uncertainty_rows,
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _setting_change_markdown(change: pd.DataFrame) -> str:
    rows = [
        [
            record["label"],
            record["mean_nats"],
            record["median_nats"],
            record["relative_pct"],
            record["share_positive_pct"],
        ]
        for record in change.to_dict(orient="records")
    ]
    return _markdown_rows(
        ["Quantity", "Mean (nats)", "Median (nats)", "Relative (%)", "Share positive (%)"],
        rows,
    )


def _write_report(
    path: Path,
    experiment_id: str,
    badge: str,
    comparisons: list[Comparison],
    performance: pd.DataFrame,
    effects: pd.DataFrame,
    figure_names: list[str],
    setting_change: pd.DataFrame | None = None,
) -> None:
    pooled = performance[performance["fold"] == "pooled"]
    lines = [
        f"# {experiment_id}",
        "",
        f"Evaluated on the **{badge.lower()}**. "
        "Classification changes are candidate minus reference, in percentage points. "
        "Uncertainty reductions are reference minus candidate, in nats: "
        "a positive reduction means the candidate is more certain.",
        "",
        "## Comparisons",
        "",
    ]
    for comparison in comparisons:
        lines.append(
            f"- **{comparison.name}**: {comparison.reference.label} "
            f"(`{comparison.reference.run_group}`) versus {comparison.candidate.label} "
            f"(`{comparison.candidate.run_group}`)."
        )
    lines += [
        "",
        "## Performance and uncertainty",
        "",
        "Pooled rows recompute accuracy, balanced accuracy, macro-F1, NLL, and Brier "
        "on the concatenated held-out images. Uncertainty means are image-weighted.",
        "",
        _markdown_table(
            pooled,
            [
                "condition",
                "accuracy_pct",
                "balanced_accuracy_pct",
                "macro_f1_pct",
                "nll_nats",
                "brier",
                "mean_total_uncertainty_nats",
                "mean_aleatoric_uncertainty_nats",
                "mean_epistemic_uncertainty_nats",
            ],
        ),
        "",
        "## Paired effects",
        "",
        "Each comparison is two narrow tables. "
        "The classification row is candidate minus reference, in percentage points. "
        "The uncertainty rows are reference minus candidate.",
        "",
        _paired_effects_markdown(effects),
        "",
    ]
    if setting_change is not None:
        lines += [
            "## Uncertainty change between settings",
            "",
            "These rows compare the candidate setting with the baseline setting. "
            "They are not the image-only versus metadata effect. "
            "dAU, dEU, and dTU are baseline minus candidate, in nats: "
            "a positive value means the candidate setting is more certain. "
            "d(dEU) is the candidate metadata EU reduction minus the baseline metadata "
            "EU reduction: a positive value means metadata reduces epistemic uncertainty "
            "by more under the candidate setting. "
            "Relative change uses the baseline mean as the denominator. "
            "Share positive is the percentage of images with a strictly positive value.",
            "",
            _setting_change_markdown(setting_change),
            "",
        ]
    lines += [
        "## Figures",
        "",
    ]
    for name in figure_names:
        lines.append(f"![{name}](figures/{name})")
        lines.append("")
    lines += [
        "## Metric conventions",
        "",
        "- Accuracy, balanced accuracy, macro-F1, precision, recall, and F1 are percentages.",
        "- Macro-F1 and balanced accuracy average over diagnoses present in the evaluated images.",
        "- Per-class precision, recall, and F1 use zero when a denominator is zero.",
        "- Negative log-likelihood floors the true-class probability at 1e-12 "
        "and does not otherwise clip it.",
        "- Relative uncertainty reduction is unavailable when the mean reference uncertainty is 0.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _comparisons(cfg: DictConfig, runs_root: Path) -> tuple[list[Condition], list[Comparison]]:
    n_folds = int(cfg.n_folds)
    split = str(cfg.eval_split)
    four_way = all(
        _as_text(cfg.get(key))
        for key in (
            "baseline_image",
            "baseline_multimodal",
            "candidate_image",
            "candidate_multimodal",
        )
    )
    single = bool(_as_text(cfg.get("reference"))) and bool(_as_text(cfg.get("candidate")))
    if four_way == bool(single):
        raise ValueError(
            "Set either reference and candidate, or all four of baseline_image, "
            "baseline_multimodal, candidate_image, and candidate_multimodal."
        )
    if four_way:
        baseline_label = _as_text(cfg.get("baseline_label")) or "Baseline"
        candidate_label = _as_text(cfg.get("candidate_label"))
        if candidate_label is None:
            raise ValueError("candidate_label is required for a baseline-versus-candidate report")
        specs = [
            ("baseline_image", f"{baseline_label} (Image-only)", _as_text(cfg.baseline_image)),
            (
                "baseline_multimodal",
                f"{baseline_label} (Image + Metadata)",
                _as_text(cfg.baseline_multimodal),
            ),
            ("candidate_image", f"{candidate_label} (Image-only)", _as_text(cfg.candidate_image)),
            (
                "candidate_multimodal",
                f"{candidate_label} (Image + Metadata)",
                _as_text(cfg.candidate_multimodal),
            ),
        ]
        loaded = {
            key: load_condition(runs_root, run_group, label, n_folds, split)
            for key, label, run_group in specs
            if run_group is not None
        }
        conditions = [loaded[key] for key, _, _ in specs]
        comparisons = [
            Comparison(
                "baseline_modality",
                f"{baseline_label} modality effect",
                loaded["baseline_image"],
                loaded["baseline_multimodal"],
            ),
            Comparison(
                "candidate_modality",
                f"{candidate_label} modality effect",
                loaded["candidate_image"],
                loaded["candidate_multimodal"],
            ),
            Comparison(
                "image_only_change",
                f"Image-only: {baseline_label} -> {candidate_label}",
                loaded["baseline_image"],
                loaded["candidate_image"],
            ),
            Comparison(
                "multimodal_change",
                f"Image + Metadata: {baseline_label} -> {candidate_label}",
                loaded["baseline_multimodal"],
                loaded["candidate_multimodal"],
            ),
        ]
        return conditions, comparisons

    reference_name = _as_text(cfg.reference)
    candidate_name = _as_text(cfg.candidate)
    assert reference_name is not None and candidate_name is not None
    reference = load_condition(
        runs_root,
        reference_name,
        _infer_label(reference_name, _as_text(cfg.get("reference_label"))),
        n_folds,
        split,
    )
    candidate = load_condition(
        runs_root,
        candidate_name,
        _infer_label(candidate_name, _as_text(cfg.get("candidate_label"))),
        n_folds,
        split,
    )
    comparison = Comparison(
        "primary", f"{reference.label} vs {candidate.label}", reference, candidate
    )
    return [reference, candidate], [comparison]


def _update_summary(summary_dir: Path, experiment_id: str, performance: pd.DataFrame) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    pooled = performance[performance["fold"] == "pooled"][
        [
            "condition",
            "n_images",
            "mean_total_uncertainty_nats",
            "mean_aleatoric_uncertainty_nats",
        ]
    ].copy()
    pooled.insert(0, "experiment_id", experiment_id)
    pooled.insert(1, "setup", pooled.pop("condition"))
    path = summary_dir / "experiment_uncertainty.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["experiment_id"] != experiment_id]
        pooled = pd.concat([existing, pooled], ignore_index=True)
    pooled = pooled.sort_values(["experiment_id", "setup"]).reset_index(drop=True)
    pooled.to_csv(path, index=False)
    plot_total_vs_aleatoric(pooled, summary_dir / "total_vs_aleatoric_uncertainty.png")


def run_analysis(cfg: DictConfig) -> Path:
    experiment_id = _as_text(cfg.get("experiment_id"))
    if experiment_id is None or "/" in experiment_id or experiment_id in {".", ".."}:
        raise ValueError("experiment_id must be a single directory name")
    split = str(cfg.eval_split)
    badge = _split_badge(split)
    runs_root = Path(cfg.runs_root)
    output = Path(cfg.analysis_root) / experiment_id
    if output.exists():
        for child in ("tables", "data", "figures"):
            target = output / child
            if target.exists():
                for path in target.iterdir():
                    if path.is_file():
                        path.unlink()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    conditions, comparisons = _comparisons(cfg, runs_root)
    paired = {comparison.comparison_id: pair_comparison(comparison) for comparison in comparisons}
    paired_frame = pd.concat(paired.values(), ignore_index=True)
    cohort = cohort_table(comparisons, paired)
    performance = performance_table(conditions)
    effects = paired_effects_table(comparisons, paired)
    setting_change = setting_uncertainty_change(
        paired, samples=int(cfg.bootstrap_samples), seed=int(cfg.bootstrap_seed)
    )
    classes = class_breakdown_table(conditions, comparisons, paired)
    history = load_validation_history(conditions)
    paired_frame.to_csv(output / "data" / "paired_images.csv", index=False)
    history.to_csv(output / "data" / "validation_history.csv", index=False)
    cohort.to_csv(output / "tables" / "cohort_and_pairing.csv", index=False)
    performance.to_csv(output / "tables" / "performance_and_uncertainty.csv", index=False)
    effects.to_csv(output / "tables" / "paired_effects.csv", index=False)
    if setting_change is not None:
        setting_change.to_csv(output / "tables" / "setting_uncertainty_change.csv", index=False)
    classes.to_csv(output / "tables" / "class_breakdown.csv", index=False)

    figure_names = []
    for comparison in comparisons:
        frame = paired[comparison.comparison_id]
        scatter_name = f"paired_eu_scatter_{comparison.comparison_id}.png"
        distribution_name = f"paired_deu_distribution_{comparison.comparison_id}.png"
        candles_name = f"paired_deu_candles_{comparison.comparison_id}.png"
        plot_paired_eu_scatter(frame, comparison.name, badge, output / "figures" / scatter_name)
        plot_deu_distribution(frame, comparison.name, badge, output / "figures" / distribution_name)
        plot_deu_candles(frame, comparison.name, badge, output / "figures" / candles_name)
        figure_names.extend([scatter_name, distribution_name, candles_name])
    plot_confusion_matrices(
        [(condition.label, _confusion(condition)) for condition in conditions],
        list(CLASSES),
        badge,
        output / "figures" / "confusion_matrices.png",
    )
    plot_effect_estimates(
        _effect_rows(comparisons, paired, int(cfg.bootstrap_samples), int(cfg.bootstrap_seed)),
        badge,
        output / "figures" / "paired_effect_estimates.png",
    )
    plot_validation_curves(history, output / "figures" / "validation_curves.png")
    figure_names.extend(
        ["confusion_matrices.png", "paired_effect_estimates.png", "validation_curves.png"]
    )
    if setting_change is not None:
        plot_setting_uncertainty_change(
            setting_change, badge, output / "figures" / "setting_uncertainty_change.png"
        )
        figure_names.append("setting_uncertainty_change.png")

    manifest = {
        "experiment_id": experiment_id,
        "eval_split": split,
        "source_revision": _git_revision(),
        "class_order": list(CLASSES),
        "metric_conventions": {
            "classification": "percent",
            "classification_change": "candidate minus reference, percentage points",
            "uncertainty": "nats, image-weighted mean",
            "uncertainty_reduction": "reference minus candidate",
            "nll_floor": 1e-12,
            "macro_average": "classes with support only",
            "zero_division": 0,
            "relative_reduction": "unavailable when the reference mean is 0",
            **(
                {
                    "setting_uncertainty_change": (
                        "dTU, dAU, and dEU are baseline minus candidate; "
                        "d_dEU is the candidate metadata EU reduction minus "
                        "the baseline metadata EU reduction"
                    )
                }
                if setting_change is not None
                else {}
            ),
        },
        "conditions": [
            {
                "label": condition.label,
                "run_group": condition.run_group,
                "folds": [str(fold.directory) for fold in condition.folds],
                "settings": _settings(condition.folds[0].directory),
            }
            for condition in conditions
        ],
        "comparisons": [
            {
                "id": comparison.comparison_id,
                "name": comparison.name,
                "reference": comparison.reference.label,
                "candidate": comparison.candidate.label,
            }
            for comparison in comparisons
        ],
    }
    OmegaConf.save(OmegaConf.create(manifest), output / "manifest.yaml", resolve=True)
    _write_report(
        output / "REPORT.md",
        experiment_id,
        badge,
        comparisons,
        performance,
        effects,
        figure_names,
        setting_change,
    )
    if split == "test":
        _update_summary(Path(cfg.analysis_root) / "summary", experiment_id, performance)
    return output

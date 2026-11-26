"""Paired, patient-level analysis of deep-ensemble experiment artifacts."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from src.analysis_plots import (
    plot_calibration,
    plot_classification,
    plot_condition_comparison,
    plot_correct_vs_incorrect,
    plot_cross_setup_fold_eu,
    plot_cross_setup_learning_curves,
    plot_ensemble_size_stability,
    plot_eu_measures_comparison,
    plot_fold_comparison,
    plot_missing_metadata_stratification,
    plot_paired_eu_scatter,
    plot_risk_coverage,
    plot_train_fraction,
    plot_training_curves,
    plot_uncertainty,
)
from src.metrics import detailed_metrics
from src.uq.deep_ensemble import uncertainty

log = logging.getLogger(__name__)

CLINICAL_BLOCK_COLUMNS = (
    "smoke",
    "drink",
    "background_father",
    "background_mother",
    "pesticide",
    "gender",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "fitspatrick",
    "diameter_1",
    "diameter_2",
)
HISTORY_METRICS = (
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "train_balanced_accuracy",
    "val_balanced_accuracy",
    "train_macro_f1",
    "val_macro_f1",
)


@dataclass
class EnsembleArtifact:
    run_dir: Path
    cfg: DictConfig
    probabilities: np.ndarray
    labels: np.ndarray
    img_ids: np.ndarray
    patient_ids: np.ndarray
    classes: np.ndarray
    seeds: np.ndarray
    logits: np.ndarray | None = None
    val_logits: np.ndarray | None = None
    val_labels: np.ndarray | None = None


def load_ensemble_artifact(run_dir: str | Path) -> EnsembleArtifact:
    path = Path(run_dir)
    with np.load(path / "test_member_predictions.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    required = {"probabilities", "labels", "img_ids", "patient_ids", "classes", "seeds"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"{path} is missing ensemble arrays: {sorted(missing)}")
    probabilities = arrays["probabilities"]
    if probabilities.ndim != 3:
        raise ValueError(f"{path} probabilities must have shape (member, image, class)")
    logits = arrays.get("logits")
    val_logits: np.ndarray | None = None
    val_labels: np.ndarray | None = None
    val_path = path / "val_member_predictions.npz"
    if val_path.exists():
        with np.load(val_path, allow_pickle=False) as val_saved:
            if "logits" in val_saved.files and "labels" in val_saved.files:
                val_logits = val_saved["logits"].copy()
                val_labels = val_saved["labels"].copy()
    return EnsembleArtifact(
        run_dir=path,
        cfg=OmegaConf.load(path / "config.yaml"),
        probabilities=probabilities,
        labels=arrays["labels"],
        img_ids=arrays["img_ids"].astype(str),
        patient_ids=arrays["patient_ids"].astype(str),
        classes=arrays["classes"].astype(str),
        seeds=arrays["seeds"],
        logits=logits,
        val_logits=val_logits,
        val_labels=val_labels,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit scalar temperature T > 0 minimizing validation NLL (no test leakage)."""
    logits_arr = np.asarray(logits, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    idx = np.arange(len(labels_arr))

    def _nll(log_t: float) -> float:
        temp = float(np.exp(log_t))
        ls = log_softmax(logits_arr / temp, axis=-1)
        return float(-ls[idx, labels_arr].mean())

    result = minimize_scalar(_nll, bounds=(-2.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def _temperature_scaled_probabilities(
    artifact: EnsembleArtifact,
) -> tuple[np.ndarray, np.ndarray] | None:
    if (
        artifact.logits is None
        or artifact.val_logits is None
        or artifact.val_labels is None
    ):
        return None
    if (
        artifact.logits.shape != artifact.probabilities.shape
        or artifact.val_logits.shape[0] != artifact.probabilities.shape[0]
    ):
        return None
    temps = np.array(
        [
            fit_temperature(artifact.val_logits[k], artifact.val_labels)
            for k in range(artifact.logits.shape[0])
        ],
        dtype=float,
    )
    scaled = np.stack(
        [softmax(artifact.logits[k] / temps[k], axis=-1) for k in range(len(temps))]
    )
    return scaled, temps


def bootstrap_mean_ci(
    values: np.ndarray, n_samples: int = 5000, seed: int = 42
) -> tuple[float, float]:
    """Percentile interval for a mean over independent patient values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a non-empty one-dimensional array")
    if n_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def variance_epistemic(member_probs: np.ndarray) -> np.ndarray:
    """Sum of label-wise ensemble variance for every image."""
    return np.var(member_probs, axis=0).sum(axis=1)


def _normalized_epistemic(epistemic: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.divide(epistemic, total, out=np.zeros_like(epistemic), where=total > 0)


def _member_disagreement_rate(member_probs: np.ndarray) -> np.ndarray:
    """Fraction of disagreeing member pairs per image, in [0, 1]."""
    n_members = member_probs.shape[0]
    if n_members < 2:
        return np.zeros(member_probs.shape[1], dtype=float)
    preds = member_probs.argmax(axis=2)  # (M, N)
    agree = (preds[:, None, :] == preds[None, :, :]).sum(axis=(0, 1)) - n_members
    total_pairs = n_members * (n_members - 1)
    return 1.0 - agree / total_pairs


def _config_value(artifact: EnsembleArtifact, path: str) -> Any:
    value = OmegaConf.select(artifact.cfg, path)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _condition_signature(cfg: DictConfig) -> str:
    payload = {
        path: (
            OmegaConf.to_container(val, resolve=True)
            if OmegaConf.is_config(val)
            else val
        )
        for path in ("split", "data", "train", "model.image_encoder", "ensemble.seeds")
        for val in (OmegaConf.select(cfg, path),)
    }
    return json.dumps(payload, sort_keys=True)


def _strip_arm_suffix(name: str) -> str:
    for suffix in ("_image_only", "_multimodal", "-image-only", "-multimodal"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_conditions(runs_root: str | Path) -> dict[str, dict[str, list[str]]]:
    """Discover completed paired deep-ensemble runs under `runs_root`."""
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {root}")

    grouped: dict[str, dict[str, dict[int, Path]]] = {}
    base_names: dict[str, list[str]] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.yaml"
        npz_path = run_dir / "test_member_predictions.npz"
        if not (cfg_path.exists() and npz_path.exists()):
            continue
        cfg = OmegaConf.load(cfg_path)
        if cfg.get("ensemble") is None:
            continue
        signature = _condition_signature(cfg)
        arm = "image" if cfg.model.get("tabular_encoder") is None else "multimodal"
        fold = int(cfg.fold)
        slot = grouped.setdefault(signature, {"image": {}, "multimodal": {}})
        if fold in slot[arm]:
            raise ValueError(
                f"duplicate {arm} fold {fold} for signature under {root}: "
                f"{slot[arm][fold]} and {run_dir}"
            )
        slot[arm][fold] = run_dir
        name = str(cfg.get("name", run_dir.name))
        base_names.setdefault(signature, []).append(_strip_arm_suffix(name))

    conditions: dict[str, dict[str, list[str]]] = {}
    used_labels: set[str] = set()
    for signature, arms in grouped.items():
        shared_folds = sorted(set(arms["image"]).intersection(arms["multimodal"]))
        if not shared_folds:
            log.warning("skipping unpaired runs for signature %s", base_names[signature][0])
            continue
        if set(arms["image"]) != set(arms["multimodal"]):
            log.warning(
                "condition %s has partial fold overlap; using shared folds %s",
                base_names[signature][0],
                shared_folds,
            )
        label = base_names[signature][0]
        if label in used_labels:
            first_cfg = OmegaConf.load(arms["image"][shared_folds[0]] / "config.yaml")
            arch = OmegaConf.select(first_cfg, "model.image_encoder.arch")
            raw_frac = float(OmegaConf.select(first_cfg, "data.train_fraction") or 1.0)
            frac = int(round(raw_frac * 100))
            label = f"{label}_{arch}_f{frac:03d}"
        used_labels.add(label)
        conditions[label] = {
            "image_run_dirs": [str(arms["image"][f]) for f in shared_folds],
            "multimodal_run_dirs": [str(arms["multimodal"][f]) for f in shared_folds],
        }

    if not conditions:
        raise ValueError(f"no paired deep-ensemble runs found under {root}")
    return conditions


def _validate_arm(artifacts: list[EnsembleArtifact], arm: str) -> dict[int, EnsembleArtifact]:
    by_fold: dict[int, EnsembleArtifact] = {}
    for artifact in artifacts:
        fold = int(artifact.cfg.fold)
        if fold in by_fold:
            raise ValueError(f"{arm} has duplicate fold {fold}")
        if len(set(artifact.img_ids)) != len(artifact.img_ids):
            raise ValueError(f"{artifact.run_dir} has duplicate image IDs")
        if artifact.probabilities.shape[0] != len(artifact.seeds):
            raise ValueError(f"{artifact.run_dir} member and seed counts disagree")
        if artifact.probabilities.shape[1] != len(artifact.img_ids):
            raise ValueError(f"{artifact.run_dir} image arrays disagree")
        if artifact.probabilities.shape[2] != len(artifact.classes):
            raise ValueError(f"{artifact.run_dir} class arrays disagree")
        by_fold[fold] = artifact
    return by_fold


def _pair_fold(
    image: EnsembleArtifact, multimodal: EnsembleArtifact
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for path in (
        "fold",
        "split",
        "data",
        "train",
        "model.image_encoder",
        "ensemble.seeds",
    ):
        if _config_value(image, path) != _config_value(multimodal, path):
            raise ValueError(
                f"paired runs disagree on {path}: {image.run_dir}, {multimodal.run_dir}"
            )
    if image.cfg.model.get("tabular_encoder") is not None:
        raise ValueError(f"image-only run has a tabular encoder: {image.run_dir}")
    if multimodal.cfg.model.get("tabular_encoder") is None:
        raise ValueError(f"multimodal run has no tabular encoder: {multimodal.run_dir}")
    if not np.array_equal(image.classes, multimodal.classes):
        raise ValueError("paired runs disagree on class order")
    if not np.array_equal(image.seeds, multimodal.seeds):
        raise ValueError("paired runs disagree on ensemble seeds")
    if set(image.img_ids) != set(multimodal.img_ids):
        raise ValueError("paired runs have different test image IDs")

    multimodal_index = pd.Index(multimodal.img_ids)
    order = multimodal_index.get_indexer(image.img_ids)
    if (order < 0).any():
        raise ValueError("failed to align paired image IDs")
    if not np.array_equal(image.labels, multimodal.labels[order]):
        raise ValueError("paired runs disagree on labels")
    if not np.array_equal(image.patient_ids, multimodal.patient_ids[order]):
        raise ValueError("paired runs disagree on patient IDs")
    return image.probabilities, multimodal.probabilities[:, order], order


def _fold_frame(
    image: EnsembleArtifact, multimodal: EnsembleArtifact
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    image_probs, multimodal_probs, order = _pair_fold(image, multimodal)
    image_mean = image_probs.mean(axis=0)
    multimodal_mean = multimodal_probs.mean(axis=0)
    image_total, image_aleatoric, image_epistemic = uncertainty(image_probs)
    multimodal_total, multimodal_aleatoric, multimodal_epistemic = uncertainty(
        multimodal_probs
    )
    image_pred = image_mean.argmax(axis=1)
    multimodal_pred = multimodal_mean.argmax(axis=1)
    classes = image.classes
    frame = pd.DataFrame(
        {
            "fold": int(image.cfg.fold),
            "patient_id": image.patient_ids,
            "img_id": image.img_ids,
            "diagnostic": classes[image.labels],
            "label": image.labels,
            "image_predicted": classes[image_pred],
            "multimodal_predicted": classes[multimodal_pred],
            "image_correct": image_pred == image.labels,
            "multimodal_correct": multimodal_pred == image.labels,
            "image_confidence": image_mean.max(axis=1),
            "multimodal_confidence": multimodal_mean.max(axis=1),
            "image_disagreement": _member_disagreement_rate(image_probs),
            "multimodal_disagreement": _member_disagreement_rate(multimodal_probs),
            "image_total": image_total,
            "multimodal_total": multimodal_total,
            "image_aleatoric": image_aleatoric,
            "multimodal_aleatoric": multimodal_aleatoric,
            "image_epistemic": image_epistemic,
            "multimodal_epistemic": multimodal_epistemic,
            "image_variance_epistemic": variance_epistemic(image_probs),
            "multimodal_variance_epistemic": variance_epistemic(multimodal_probs),
            "image_normalized_epistemic": _normalized_epistemic(
                image_epistemic, image_total
            ),
            "multimodal_normalized_epistemic": _normalized_epistemic(
                multimodal_epistemic, multimodal_total
            ),
        }
    )
    image_ts = _temperature_scaled_probabilities(image)
    multimodal_ts = _temperature_scaled_probabilities(multimodal)
    if image_ts is not None and multimodal_ts is not None:
        img_ts_probs, img_temps = image_ts
        mm_ts_probs_raw, mm_temps = multimodal_ts
        mm_ts_probs = mm_ts_probs_raw[:, order]
        img_ts_tu, img_ts_au, img_ts_eu = uncertainty(img_ts_probs)
        mm_ts_tu, mm_ts_au, mm_ts_eu = uncertainty(mm_ts_probs)
        idx = np.arange(len(image.labels))
        img_ts_mean = img_ts_probs.mean(axis=0)
        mm_ts_mean = mm_ts_probs.mean(axis=0)
        frame["image_ts_total"] = img_ts_tu
        frame["multimodal_ts_total"] = mm_ts_tu
        frame["image_ts_aleatoric"] = img_ts_au
        frame["multimodal_ts_aleatoric"] = mm_ts_au
        frame["image_ts_epistemic"] = img_ts_eu
        frame["multimodal_ts_epistemic"] = mm_ts_eu
        frame["dts_epistemic"] = img_ts_eu - mm_ts_eu
        frame["image_ts_nll"] = -np.log(np.clip(img_ts_mean[idx, image.labels], 1e-12, 1.0))
        frame["multimodal_ts_nll"] = -np.log(
            np.clip(mm_ts_mean[idx, image.labels], 1e-12, 1.0)
        )
        frame["image_mean_temp"] = float(img_temps.mean())
        frame["multimodal_mean_temp"] = float(mm_temps.mean())
    for index, name in enumerate(classes):
        frame[f"image_p_{name}"] = image_mean[:, index]
        frame[f"multimodal_p_{name}"] = multimodal_mean[:, index]
    for suffix in ("epistemic", "variance_epistemic", "normalized_epistemic"):
        frame[f"d{suffix}"] = frame[f"image_{suffix}"] - frame[f"multimodal_{suffix}"]
    return frame, image_probs, multimodal_probs


def _paired_test(image_values: np.ndarray, multimodal_values: np.ndarray) -> float:
    difference = np.asarray(image_values) - np.asarray(multimodal_values)
    if np.allclose(difference, 0):
        return 1.0
    return float(wilcoxon(image_values, multimodal_values, alternative="two-sided").pvalue)


def _summarize_subset(
    frame: pd.DataFrame,
    stratum_type: str,
    stratum_value: str,
    bootstrap_samples: int,
    seed: int,
    group_col: str = "patient_id",
) -> list[dict[str, Any]]:
    measures = {
        "entropy_epistemic": ("image_epistemic", "multimodal_epistemic"),
        "variance_epistemic": (
            "image_variance_epistemic",
            "multimodal_variance_epistemic",
        ),
        "normalized_epistemic": (
            "image_normalized_epistemic",
            "multimodal_normalized_epistemic",
        ),
    }
    records = []
    for measure, (image_column, multimodal_column) in measures.items():
        grouped = frame.groupby(group_col)[[image_column, multimodal_column]].mean()
        difference = grouped[image_column] - grouped[multimodal_column]
        low, high = bootstrap_mean_ci(difference.to_numpy(), bootstrap_samples, seed)
        records.append(
            {
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "measure": measure,
                "n_images": len(frame),
                "n_patients": len(grouped),
                "image_mean": float(grouped[image_column].mean()),
                "multimodal_mean": float(grouped[multimodal_column].mean()),
                "mean_deu": float(difference.mean()),
                "deu_ci_low": low,
                "deu_ci_high": high,
                "wilcoxon_p": _paired_test(
                    grouped[image_column].to_numpy(),
                    grouped[multimodal_column].to_numpy(),
                ),
            }
        )
    return records


def summarize_uncertainty(
    frame: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    records = _summarize_subset(frame, "overall", "all", bootstrap_samples, seed)
    for diagnostic, subset in frame.groupby("diagnostic"):
        records.extend(
            _summarize_subset(
                subset, "class", str(diagnostic), bootstrap_samples, seed
            )
        )
    for column in ("image_correct", "multimodal_correct"):
        for value, subset in frame.groupby(column):
            records.extend(
                _summarize_subset(
                    subset,
                    column,
                    str(bool(value)).lower(),
                    bootstrap_samples,
                    seed,
                )
            )
    return pd.DataFrame(records)


def _classification_metrics(frame: pd.DataFrame, arm: str, classes: list[str]) -> dict:
    labels = frame["label"].to_numpy()
    probs = frame[[f"{arm}_p_{name}" for name in classes]].to_numpy()
    return detailed_metrics(labels, probs.argmax(axis=1), classes, probs)


def _error_detection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ("image", "multimodal"):
        is_error = (~frame[f"{arm}_correct"]).astype(int).to_numpy()
        score_map = {
            "epistemic": frame[f"{arm}_epistemic"].to_numpy(),
            "aleatoric": frame[f"{arm}_aleatoric"].to_numpy(),
            "total": frame[f"{arm}_total"].to_numpy(),
            "1-confidence": 1.0 - frame[f"{arm}_confidence"].to_numpy(),
        }
        for score_name, values in score_map.items():
            auroc = (
                float(roc_auc_score(is_error, values))
                if len(np.unique(is_error)) == 2
                else None
            )
            rows.append(
                {
                    "arm": arm,
                    "score": score_name,
                    "error_rate": float(is_error.mean()),
                    "mean_correct": (
                        float(values[is_error == 0].mean()) if (is_error == 0).any() else None
                    ),
                    "mean_incorrect": (
                        float(values[is_error == 1].mean()) if (is_error == 1).any() else None
                    ),
                    "error_auroc": auroc,
                }
            )
    return pd.DataFrame(rows)


def _risk_coverage_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverages = np.linspace(0.1, 1.0, 19)
    curve_rows, aurc_rows = [], []
    for arm in ("image", "multimodal"):
        errors = (~frame[f"{arm}_correct"]).astype(float).to_numpy()
        score_map = {
            "epistemic": frame[f"{arm}_epistemic"].to_numpy(),
            "aleatoric": frame[f"{arm}_aleatoric"].to_numpy(),
            "total": frame[f"{arm}_total"].to_numpy(),
            "1-confidence": 1.0 - frame[f"{arm}_confidence"].to_numpy(),
        }
        for score_name, scores in score_map.items():
            order = np.argsort(scores, kind="mergesort")
            sorted_errors = errors[order]
            n = len(sorted_errors)
            ks = np.arange(1, n + 1)
            cum_risk = np.cumsum(sorted_errors) / ks
            full_cov = ks / n
            aurc = float(np.trapezoid(cum_risk, full_cov))
            aurc_rows.append({"arm": arm, "score": score_name, "aurc": aurc})
            for cov in coverages:
                k = max(1, int(np.ceil(cov * n)))
                curve_rows.append(
                    {
                        "arm": arm,
                        "score": score_name,
                        "coverage": float(k / n),
                        "risk": float(cum_risk[k - 1]),
                    }
                )
    return pd.DataFrame(curve_rows), pd.DataFrame(aurc_rows)


def _aggregate_training_curves(
    image: dict[int, EnsembleArtifact], multimodal: dict[int, EnsembleArtifact]
) -> pd.DataFrame | None:
    frames = []
    for arm, by_fold in (("image", image), ("multimodal", multimodal)):
        for fold, artifact in sorted(by_fold.items()):
            for seed in artifact.seeds:
                history_path = artifact.run_dir / "members" / f"seed_{int(seed)}" / "history.csv"
                if not history_path.exists():
                    continue
                hist = pd.read_csv(history_path)
                hist["arm"] = arm
                hist["fold"] = fold
                hist["seed"] = int(seed)
                frames.append(hist)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for (arm, epoch), group in combined.groupby(["arm", "epoch"], sort=True):
        record: dict[str, Any] = {
            "arm": str(arm),
            "epoch": int(epoch),
            "n_runs": len(group),
        }
        for metric in HISTORY_METRICS:
            if metric in group:
                vals = group[metric].dropna()
                record[f"{metric}_mean"] = float(vals.mean()) if len(vals) else None
                record[f"{metric}_std"] = float(vals.std(ddof=0)) if len(vals) else None
        rows.append(record)
    return pd.DataFrame(rows)


def _per_fold_summary(
    condition: str,
    frame: pd.DataFrame,
    classes: list[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for fold, sub in frame.groupby("fold", sort=True):
        patient = sub.groupby("patient_id")[
            [
                "image_total",
                "multimodal_total",
                "image_aleatoric",
                "multimodal_aleatoric",
                "image_epistemic",
                "multimodal_epistemic",
                "image_variance_epistemic",
                "multimodal_variance_epistemic",
                "image_normalized_epistemic",
                "multimodal_normalized_epistemic",
            ]
        ].mean()
        diff = patient["image_epistemic"] - patient["multimodal_epistemic"]
        low, high = bootstrap_mean_ci(diff.to_numpy(), bootstrap_samples, seed)
        m_img = _classification_metrics(sub, "image", classes)
        m_mm = _classification_metrics(sub, "multimodal", classes)
        rows.append(
            {
                "condition": condition,
                "fold": int(fold),
                "n_images": len(sub),
                "n_patients": len(patient),
                "mean_image_total": float(patient["image_total"].mean()),
                "mean_multimodal_total": float(patient["multimodal_total"].mean()),
                "mean_image_aleatoric": float(patient["image_aleatoric"].mean()),
                "mean_multimodal_aleatoric": float(patient["multimodal_aleatoric"].mean()),
                "mean_image_epistemic": float(patient["image_epistemic"].mean()),
                "mean_multimodal_epistemic": float(patient["multimodal_epistemic"].mean()),
                "mean_image_variance_epistemic": float(
                    patient["image_variance_epistemic"].mean()
                ),
                "mean_multimodal_variance_epistemic": float(
                    patient["multimodal_variance_epistemic"].mean()
                ),
                "mean_image_normalized_epistemic": float(
                    patient["image_normalized_epistemic"].mean()
                ),
                "mean_multimodal_normalized_epistemic": float(
                    patient["multimodal_normalized_epistemic"].mean()
                ),
                "mean_deu": float(diff.mean()),
                "deu_ci_low": low,
                "deu_ci_high": high,
                "wilcoxon_p": _paired_test(
                    patient["image_epistemic"].to_numpy(),
                    patient["multimodal_epistemic"].to_numpy(),
                ),
                "image_accuracy": m_img["accuracy"],
                "multimodal_accuracy": m_mm["accuracy"],
                "image_balanced_accuracy": m_img["balanced_accuracy"],
                "multimodal_balanced_accuracy": m_mm["balanced_accuracy"],
                "image_macro_f1": m_img["macro_f1"],
                "multimodal_macro_f1": m_mm["macro_f1"],
                "image_weighted_f1": m_img["weighted_f1"],
                "multimodal_weighted_f1": m_mm["weighted_f1"],
                "image_mel_recall": m_img.get("mel_recall"),
                "multimodal_mel_recall": m_mm.get("mel_recall"),
                "image_macro_ovr_auroc": m_img.get("macro_ovr_auroc"),
                "multimodal_macro_ovr_auroc": m_mm.get("macro_ovr_auroc"),
                "image_nll": m_img["negative_log_likelihood"],
                "multimodal_nll": m_mm["negative_log_likelihood"],
                "image_brier": m_img["multiclass_brier"],
                "multimodal_brier": m_mm["multiclass_brier"],
                "image_ece": m_img["ece"],
                "multimodal_ece": m_mm["ece"],
            }
        )
    return pd.DataFrame(rows)


def _ensemble_size_stability(
    paired_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evaluate matched member subsets of size M = 2..n_members post-hoc."""
    n_members = paired_folds[0][0].shape[0]
    patient_ids = np.concatenate([pids for _, _, pids in paired_folds])
    image_all = np.concatenate([img for img, _, _ in paired_folds], axis=1)
    mm_all = np.concatenate([mm for _, mm, _ in paired_folds], axis=1)

    rows = []
    for size in range(2, n_members + 1):
        img_means, mm_means, deu_means = [], [], []
        combos = list(itertools.combinations(range(n_members), size))
        for subset in combos:
            idx = list(subset)
            _, _, img_eu = uncertainty(image_all[idx])
            _, _, mm_eu = uncertainty(mm_all[idx])
            df = pd.DataFrame(
                {"patient_id": patient_ids, "img_eu": img_eu, "mm_eu": mm_eu}
            ).groupby("patient_id")[["img_eu", "mm_eu"]].mean()
            img_means.append(float(df["img_eu"].mean()))
            mm_means.append(float(df["mm_eu"].mean()))
            deu_means.append(float((df["img_eu"] - df["mm_eu"]).mean()))
        rows.append(
            {
                "n_members": size,
                "n_combinations": len(combos),
                "image_eu_mean": float(np.mean(img_means)),
                "image_eu_std": float(np.std(img_means, ddof=0)),
                "multimodal_eu_mean": float(np.mean(mm_means)),
                "multimodal_eu_std": float(np.std(mm_means, ddof=0)),
                "mean_deu_mean": float(np.mean(deu_means)),
                "mean_deu_std": float(np.std(deu_means, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _missing_metadata_analysis(
    frame: pd.DataFrame,
    metadata_path: Path | None,
    classes: list[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    meta = pd.read_csv(metadata_path, dtype={"patient_id": str, "img_id": str})
    meta.columns = meta.columns.str.strip()
    if "img_id" not in meta:
        return None
    block_cols = [c for c in CLINICAL_BLOCK_COLUMNS if c in meta.columns]
    cols_to_merge = ["img_id"]
    if "lesion_id" in meta.columns:
        cols_to_merge.append("lesion_id")
    if block_cols:
        meta["clinical_block_missing"] = meta[block_cols].isna().all(axis=1)
        cols_to_merge.append("clinical_block_missing")
    merged = frame.merge(meta[cols_to_merge], on="img_id", how="left", validate="one_to_one")

    rows = []

    def add_row(name: str, subset: pd.DataFrame, group_col: str) -> None:
        if subset.empty:
            return
        grouped = subset.groupby(group_col)[["image_epistemic", "multimodal_epistemic"]].mean()
        diff = grouped["image_epistemic"] - grouped["multimodal_epistemic"]
        low, high = bootstrap_mean_ci(diff.to_numpy(), bootstrap_samples, seed)
        m_img = _classification_metrics(subset, "image", classes)
        m_mm = _classification_metrics(subset, "multimodal", classes)
        rows.append(
            {
                "stratum": name,
                "unit": group_col,
                "n_images": len(subset),
                "n_units": len(grouped),
                "image_eu": float(grouped["image_epistemic"].mean()),
                "multimodal_eu": float(grouped["multimodal_epistemic"].mean()),
                "mean_deu": float(diff.mean()),
                "deu_ci_low": low,
                "deu_ci_high": high,
                "wilcoxon_p": _paired_test(
                    grouped["image_epistemic"].to_numpy(),
                    grouped["multimodal_epistemic"].to_numpy(),
                ),
                "image_balanced_accuracy": m_img["balanced_accuracy"],
                "multimodal_balanced_accuracy": m_mm["balanced_accuracy"],
            }
        )

    add_row("all (patient-level)", merged, "patient_id")
    add_row("all (image-level)", merged, "img_id")
    if "lesion_id" in merged.columns and merged["lesion_id"].notna().all():
        merged["patient_lesion_id"] = (
            merged["patient_id"].astype(str) + "::" + merged["lesion_id"].astype(str)
        )
        add_row("all (lesion-level)", merged, "patient_lesion_id")
    if "clinical_block_missing" in merged.columns:
        complete = merged[merged["clinical_block_missing"] == False]  # noqa: E712
        missing = merged[merged["clinical_block_missing"] == True]  # noqa: E712
        add_row("clinical_block_present", complete, "patient_id")
        add_row("clinical_block_missing", missing, "patient_id")
    return pd.DataFrame(rows) if rows else None


def _eu_measures_agreement(condition_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    pairs = (
        ("entropy_vs_variance", "depistemic", "dvariance_epistemic"),
        ("entropy_vs_normalized", "depistemic", "dnormalized_epistemic"),
    )
    for condition, frame in condition_frames.items():
        patient = frame.groupby("patient_id")[
            ["depistemic", "dvariance_epistemic", "dnormalized_epistemic"]
        ].mean()
        for pair_name, col_a, col_b in pairs:
            a = patient[col_a].to_numpy()
            b = patient[col_b].to_numpy()
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                rho, pval = np.nan, np.nan
            else:
                res = spearmanr(a, b)
                rho, pval = float(res.statistic), float(res.pvalue)
            sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
            rows.append(
                {
                    "condition": condition,
                    "pair": pair_name,
                    "n_patients": len(patient),
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "patient_sign_agreement": sign_agree,
                }
            )
    return pd.DataFrame(rows)


def _attach_budget_regime(comparison: pd.DataFrame) -> pd.DataFrame:
    """Tag each condition as full, fixed_epochs, step_matched, or custom epochs."""
    out = comparison.copy()
    regimes: list[str] = []
    anchor_epochs: dict[str, int] = {}
    for arch, group in out.groupby("architecture"):
        full = group[np.isclose(group["train_fraction"], 1.0)]
        if not full.empty and "epochs" in full.columns:
            anchor_epochs[str(arch)] = int(full.iloc[0]["epochs"])
    for _, row in out.iterrows():
        frac = float(row["train_fraction"])
        epochs = int(row.get("epochs", 0) or 0)
        base_epochs = anchor_epochs.get(str(row["architecture"]), 0)
        if np.isclose(frac, 1.0):
            regimes.append("full")
        elif base_epochs > 0 and epochs == base_epochs:
            regimes.append("fixed_epochs")
        elif base_epochs > 0 and abs(epochs * frac - base_epochs) <= 1.0:
            regimes.append("step_matched")
        else:
            regimes.append(f"epochs_{epochs}" if epochs > 0 else "subset")
    out["budget_regime"] = regimes
    return out


def _positive_control_summary(
    comparison: pd.DataFrame,
    condition_frames: dict[str, pd.DataFrame],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame | None:
    rows = []
    tagged = (
        comparison
        if "budget_regime" in comparison.columns
        else _attach_budget_regime(comparison)
    )
    for arch, group in tagged.groupby("architecture"):
        full_rows = group[np.isclose(group["train_fraction"], 1.0)]
        if full_rows.empty:
            continue
        high_cond = str(full_rows.iloc[0]["condition"])
        high_frame = condition_frames[high_cond]
        subset_rows = group[~np.isclose(group["train_fraction"], 1.0)]
        if subset_rows.empty:
            continue
        for regime, reg_group in subset_rows.groupby("budget_regime"):
            by_frac = {
                round(float(row["train_fraction"]), 2): str(row["condition"])
                for _, row in reg_group.iterrows()
            }
            if 0.25 not in by_frac:
                continue
            low_frame = condition_frames[by_frac[0.25]]
            shared_folds = sorted(
                set(low_frame["fold"]).intersection(high_frame["fold"])
            )
            if not shared_folds:
                continue
            has_ts = (
                "image_ts_epistemic" in low_frame.columns
                and "image_ts_epistemic" in high_frame.columns
            )
            cols = ["image_epistemic", "multimodal_epistemic"]
            if has_ts:
                cols += ["image_ts_epistemic", "multimodal_ts_epistemic"]
            low_pat = (
                low_frame[low_frame["fold"].isin(shared_folds)]
                .groupby("patient_id")[cols]
                .mean()
            )
            high_pat = (
                high_frame[high_frame["fold"].isin(shared_folds)]
                .groupby("patient_id")[cols]
                .mean()
            )
            common_patients = low_pat.index.intersection(high_pat.index)
            if len(common_patients) == 0:
                continue
            mid_pat = None
            if 0.5 in by_frac:
                mid_frame = condition_frames[by_frac[0.5]]
                mid_cols = [c for c in cols if c in mid_frame.columns]
                mid_pat = (
                    mid_frame[mid_frame["fold"].isin(shared_folds)]
                    .groupby("patient_id")[mid_cols]
                    .mean()
                    .reindex(common_patients)
                )
            for arm in ("image", "multimodal"):
                col = f"{arm}_epistemic"
                e25 = low_pat.loc[common_patients, col].to_numpy()
                e100 = high_pat.loc[common_patients, col].to_numpy()
                e50_mean = float(mid_pat[col].mean()) if mid_pat is not None else None
                delta = e25 - e100  # positive => EU dropped with more training data
                ci_low, ci_high = bootstrap_mean_ci(delta, bootstrap_samples, seed)
                monotonic = (
                    (float(e25.mean()) > e50_mean > float(e100.mean()))
                    if e50_mean is not None
                    else float(e25.mean()) > float(e100.mean())
                )
                passed = bool(monotonic and ci_low > 0)
                status = (
                    "PASS"
                    if passed
                    else ("FAIL" if len(shared_folds) >= 5 else "UNCLEAR_EXPAND_FOLDS")
                )
                row_entry: dict[str, Any] = {
                    "architecture": str(arch),
                    "budget_regime": str(regime),
                    "arm": arm,
                    "shared_folds": ",".join(str(f) for f in shared_folds),
                    "n_patients": len(common_patients),
                    "eu_025": float(e25.mean()),
                    "eu_050": e50_mean,
                    "eu_100": float(e100.mean()),
                    "eu_drop_025_to_100": float(delta.mean()),
                    "drop_ci_low": ci_low,
                    "drop_ci_high": ci_high,
                    "wilcoxon_p": _paired_test(e25, e100),
                    "monotonic_decrease": monotonic,
                    "status": status,
                }
                if has_ts:
                    ts_col = f"{arm}_ts_epistemic"
                    te25 = low_pat.loc[common_patients, ts_col].to_numpy()
                    te100 = high_pat.loc[common_patients, ts_col].to_numpy()
                    te50_mean = (
                        float(mid_pat[ts_col].mean())
                        if mid_pat is not None and ts_col in mid_pat.columns
                        else None
                    )
                    ts_delta = te25 - te100
                    ts_lo, ts_hi = bootstrap_mean_ci(ts_delta, bootstrap_samples, seed)
                    ts_mono = (
                        (float(te25.mean()) > te50_mean > float(te100.mean()))
                        if te50_mean is not None
                        else float(te25.mean()) > float(te100.mean())
                    )
                    ts_passed = bool(ts_mono and ts_lo > 0)
                    row_entry.update(
                        {
                            "ts_eu_025": float(te25.mean()),
                            "ts_eu_050": te50_mean,
                            "ts_eu_100": float(te100.mean()),
                            "ts_eu_drop_025_to_100": float(ts_delta.mean()),
                            "ts_drop_ci_low": ts_lo,
                            "ts_drop_ci_high": ts_hi,
                            "ts_wilcoxon_p": _paired_test(te25, te100),
                            "ts_monotonic_decrease": ts_mono,
                            "ts_status": (
                                "PASS"
                                if ts_passed
                                else (
                                    "FAIL"
                                    if len(shared_folds) >= 5
                                    else "UNCLEAR_EXPAND_FOLDS"
                                )
                            ),
                        }
                    )
                rows.append(row_entry)
    return pd.DataFrame(rows) if rows else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_summary_tables_md(
    output_dir: Path,
    comparison: pd.DataFrame,
    all_folds: pd.DataFrame,
    condition_summaries: dict[str, pd.DataFrame],
    condition_missing: dict[str, pd.DataFrame],
    positive_control: pd.DataFrame | None,
    condition_frames: dict[str, pd.DataFrame],
) -> None:
    sections: list[str] = ["# Paired Epistemic Uncertainty Experiment Summary"]

    headers = [
        "Condition",
        "Arch",
        "Frac",
        "Epochs",
        "Folds",
        "Arm",
        "Acc",
        "BalAcc",
        "MacroF1",
        "WtF1",
        "MEL Rec",
        "AUROC",
        "NLL",
        "Brier",
        "ECE",
        "TU",
        "AU",
        "EU (MI)",
        "EU (Var)",
        "EU/TU",
        "dEU [95% CI]",
        "p",
    ]
    rows: list[list[str]] = []
    for _, r in comparison.iterrows():
        deu_str = f"{_fmt(r['mean_deu'])} [{_fmt(r['deu_ci_low'])}, {_fmt(r['deu_ci_high'])}]"
        for arm in ("image", "multimodal"):
            rows.append(
                [
                    str(r["condition"]),
                    str(r["architecture"]),
                    _fmt(r["train_fraction"], 2),
                    str(int(r.get("epochs", 0) or 0)),
                    str(r["folds"]),
                    arm,
                    _fmt(r[f"{arm}_accuracy"], 3),
                    _fmt(r[f"{arm}_balanced_accuracy"], 3),
                    _fmt(r[f"{arm}_macro_f1"], 3),
                    _fmt(r[f"{arm}_weighted_f1"], 3),
                    _fmt(r.get(f"{arm}_mel_recall"), 3),
                    _fmt(r.get(f"{arm}_macro_ovr_auroc"), 3),
                    _fmt(r[f"{arm}_nll"], 3),
                    _fmt(r[f"{arm}_brier"], 3),
                    _fmt(r[f"{arm}_ece"], 3),
                    _fmt(r[f"mean_{arm}_total"], 4),
                    _fmt(r[f"mean_{arm}_aleatoric"], 4),
                    _fmt(r[f"mean_{arm}_epistemic"], 4),
                    _fmt(r[f"mean_{arm}_variance_epistemic"], 4),
                    _fmt(r[f"mean_{arm}_normalized_epistemic"], 3),
                    deu_str if arm == "multimodal" else "-",
                    _fmt(r["wilcoxon_p"], 4) if arm == "multimodal" else "-",
                ]
            )
    sections.append("## 1. Overall Setup Comparison\n\n" + _markdown_table(headers, rows))

    if (
        "mean_ts_deu" in comparison.columns
        and comparison["mean_ts_deu"].notna().any()
    ):
        ts_headers = [
            "Condition",
            "Frac",
            "Epochs",
            "Regime",
            "T(img)",
            "T(mm)",
            "TS NLL(img)",
            "TS NLL(mm)",
            "TS EU(img)",
            "TS EU(mm)",
            "TS dEU [95% CI]",
            "TS p",
        ]
        ts_rows: list[list[str]] = []
        for _, r in comparison.iterrows():
            if pd.isna(r.get("mean_ts_deu")):
                continue
            ts_ci = (
                f"{_fmt(r['mean_ts_deu'])} "
                f"[{_fmt(r['ts_deu_ci_low'])}, {_fmt(r['ts_deu_ci_high'])}]"
            )
            ts_rows.append(
                [
                    str(r["condition"]),
                    _fmt(r["train_fraction"], 2),
                    str(int(r.get("epochs", 0) or 0)),
                    str(r.get("budget_regime", "n/a")),
                    _fmt(r.get("image_mean_temp"), 2),
                    _fmt(r.get("multimodal_mean_temp"), 2),
                    _fmt(r.get("image_ts_nll"), 3),
                    _fmt(r.get("multimodal_ts_nll"), 3),
                    _fmt(r.get("mean_image_ts_epistemic"), 4),
                    _fmt(r.get("mean_multimodal_ts_epistemic"), 4),
                    ts_ci,
                    _fmt(r.get("ts_wilcoxon_p"), 4),
                ]
            )
        if ts_rows:
            sections.append(
                "## 1b. Post-Hoc Validation Temperature-Scaled Epistemic Uncertainty\n\n"
                + _markdown_table(ts_headers, ts_rows)
            )

    fold_headers = [
        "Condition",
        "Fold",
        "N pat",
        "EU(img)",
        "EU(mm)",
        "dEU [95% CI]",
        "Wilcoxon p",
        "BalAcc(img)",
        "BalAcc(mm)",
        "MacroF1(img)",
        "MacroF1(mm)",
    ]
    fold_rows = [
        [
            str(r["condition"]),
            str(int(r["fold"])),
            str(int(r["n_patients"])),
            _fmt(r["mean_image_epistemic"]),
            _fmt(r["mean_multimodal_epistemic"]),
            f"{_fmt(r['mean_deu'])} [{_fmt(r['deu_ci_low'])}, {_fmt(r['deu_ci_high'])}]",
            _fmt(r["wilcoxon_p"]),
            _fmt(r["image_balanced_accuracy"], 3),
            _fmt(r["multimodal_balanced_accuracy"], 3),
            _fmt(r["image_macro_f1"], 3),
            _fmt(r["multimodal_macro_f1"], 3),
        ]
        for _, r in all_folds.iterrows()
    ]
    sections.append(
        "## 2. Per-Fold EU and Classification\n\n"
        + _markdown_table(fold_headers, fold_rows)
    )

    strat_headers = [
        "Condition",
        "Stratum",
        "Value",
        "N pat",
        "EU(img)",
        "EU(mm)",
        "dEU [95% CI]",
        "p",
    ]
    strat_rows: list[list[str]] = []
    for cond, summary in condition_summaries.items():
        mi = summary[
            (summary["measure"] == "entropy_epistemic")
            & (summary["stratum_type"] != "overall")
        ]
        for _, r in mi.iterrows():
            strat_rows.append(
                [
                    cond,
                    str(r["stratum_type"]),
                    str(r["stratum_value"]),
                    str(int(r["n_patients"])),
                    _fmt(r["image_mean"]),
                    _fmt(r["multimodal_mean"]),
                    f"{_fmt(r['mean_deu'])} [{_fmt(r['deu_ci_low'])}, {_fmt(r['deu_ci_high'])}]",
                    _fmt(r["wilcoxon_p"]),
                ]
            )
    if strat_rows:
        sections.append(
            "## 3. Stratified dEU (Class and Correctness)\n\n"
            + _markdown_table(strat_headers, strat_rows)
        )

    if condition_missing:
        miss_headers = [
            "Condition",
            "Stratum",
            "Unit",
            "N units",
            "EU(img)",
            "EU(mm)",
            "dEU [95% CI]",
            "BalAcc(img)",
            "BalAcc(mm)",
        ]
        miss_rows: list[list[str]] = []
        for cond, df in condition_missing.items():
            for _, r in df.iterrows():
                ci_text = (
                    f"{_fmt(r['mean_deu'])} "
                    f"[{_fmt(r['deu_ci_low'])}, {_fmt(r['deu_ci_high'])}]"
                )
                miss_rows.append(
                    [
                        cond,
                        str(r["stratum"]),
                        str(r["unit"]),
                        str(int(r["n_units"])),
                        _fmt(r["image_eu"]),
                        _fmt(r["multimodal_eu"]),
                        ci_text,
                        _fmt(r["image_balanced_accuracy"], 3),
                        _fmt(r["multimodal_balanced_accuracy"], 3),
                    ]
                )
        sections.append(
            "## 4. Clinical Block Completeness & Aggregation Unit\n\n"
            + _markdown_table(miss_headers, miss_rows)
        )

    decision_lines = ["## 5. Automated Decision Rules & Quality Flags"]
    for _, r in comparison.iterrows():
        for arm in ("image", "multimodal"):
            if r[f"{arm}_balanced_accuracy"] < 0.50:
                decision_lines.append(
                    f"- **Classification flag ({r['condition']} / {arm})**: "
                    f"balanced accuracy is low ({r[f'{arm}_balanced_accuracy']:.3f}); "
                    "interpret EU cautiously alongside model quality."
                )

    full_by_arch = comparison[np.isclose(comparison["train_fraction"], 1.0)]
    r18_rows = full_by_arch[full_by_arch["architecture"] == "resnet18"]
    r50_rows = full_by_arch[full_by_arch["architecture"] == "resnet50"]
    if not r18_rows.empty and not r50_rows.empty:
        r18_cond = str(r18_rows.iloc[0]["condition"])
        r50_cond = str(r50_rows.iloc[0]["condition"])
        shared = sorted(
            set(condition_frames[r18_cond]["fold"]).intersection(
                condition_frames[r50_cond]["fold"]
            )
        )
        if shared:
            r18_sub = all_folds[
                (all_folds["condition"] == r18_cond) & (all_folds["fold"].isin(shared))
            ]
            r50_sub = all_folds[
                (all_folds["condition"] == r50_cond) & (all_folds["fold"].isin(shared))
            ]
            bal_diff_img = float(
                r50_sub["image_balanced_accuracy"].mean()
                - r18_sub["image_balanced_accuracy"].mean()
            )
            bal_diff_mm = float(
                r50_sub["multimodal_balanced_accuracy"].mean()
                - r18_sub["multimodal_balanced_accuracy"].mean()
            )
            deu_r18 = float(r18_sub["mean_deu"].mean())
            deu_r50 = float(r50_sub["mean_deu"].mean())
            sign_flip = np.sign(deu_r18) != np.sign(deu_r50)
            expand_r50 = (
                abs(bal_diff_img) > 0.02 or abs(bal_diff_mm) > 0.02 or bool(sign_flip)
            )
            decision_lines.append(
                f"- **Experiment 6 (ResNet18 vs ResNet50 on shared folds {shared})**: "
                f"image BalAcc delta = {bal_diff_img:+.3f}, multimodal BalAcc delta = "
                f"{bal_diff_mm:+.3f}, mean dEU R18={deu_r18:.4f} vs R50={deu_r50:.4f} -> "
                + (
                    "**EXPAND ResNet50 to all 5 folds** (threshold triggered)."
                    if expand_r50
                    else "**KEEP ResNet18** (within 2 pp BalAcc and same dEU sign)."
                )
            )

    if positive_control is not None and not positive_control.empty:
        for _, r in positive_control.iterrows():
            regime = r.get("budget_regime", "subset")
            ts_note = ""
            if "ts_eu_025" in r and not pd.isna(r.get("ts_eu_025")):
                ts_note = (
                    f" | TS: EU(25%)={_fmt(r['ts_eu_025'])}, "
                    f"EU(50%)={_fmt(r['ts_eu_050'])}, EU(100%)={_fmt(r['ts_eu_100'])}, "
                    f"drop={_fmt(r['ts_eu_drop_025_to_100'])} "
                    f"[{_fmt(r['ts_drop_ci_low'])}, {_fmt(r['ts_drop_ci_high'])}] "
                    f"-> **{r['ts_status']}**"
                )
            decision_lines.append(
                f"- **Experiment 4 Positive Control ({r['architecture']} / {regime} / "
                f"{r['arm']}, folds {r['shared_folds']})**: "
                f"EU(25%)={_fmt(r['eu_025'])}, "
                f"EU(50%)={_fmt(r['eu_050'])}, EU(100%)={_fmt(r['eu_100'])}, "
                f"drop={_fmt(r['eu_drop_025_to_100'])} "
                f"[{_fmt(r['drop_ci_low'])}, {_fmt(r['drop_ci_high'])}], "
                f"monotonic={r['monotonic_decrease']} -> **{r['status']}**{ts_note}."
            )

    sections.append("\n".join(decision_lines))
    (output_dir / "summary_tables.md").write_text("\n\n".join(sections) + "\n")


class PairedAnalysis:
    """Run one or more paired conditions listed in a manifest or discovered under `runs_root`."""

    def __init__(
        self,
        manifest: str | None = None,
        runs_root: str | None = None,
        metadata_csv: str | None = None,
        output_dir: str = "outputs/analysis/paired",
        bootstrap_samples: int = 5000,
        seed: int = 42,
    ) -> None:
        self.manifest = Path(manifest) if manifest else None
        self.runs_root = Path(runs_root) if runs_root else None
        self.metadata_csv = Path(metadata_csv) if metadata_csv else None
        self.output_dir = Path(output_dir)
        self.bootstrap_samples = int(bootstrap_samples)
        self.seed = int(seed)

    def _load_conditions(self) -> dict[str, Any]:
        if self.manifest is not None:
            content = json.loads(self.manifest.read_text())
            conditions = content.get("conditions")
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError("analysis manifest needs a non-empty conditions object")
            return conditions
        if self.runs_root is not None:
            return discover_conditions(self.runs_root)
        raise ValueError("provide either analysis.manifest or analysis.runs_root")

    def _resolve_metadata_path(self, artifact: EnsembleArtifact) -> Path | None:
        if self.metadata_csv is not None and self.metadata_csv.exists():
            return self.metadata_csv
        cfg_meta = OmegaConf.select(artifact.cfg, "data.metadata_csv")
        if cfg_meta:
            candidate = Path(str(cfg_meta))
            if candidate.exists():
                return candidate
        return None

    def _analyze_condition(
        self, label: str, spec: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        image_paths = spec.get("image_run_dirs", [])
        multimodal_paths = spec.get("multimodal_run_dirs", [])
        if not image_paths or not multimodal_paths:
            raise ValueError(f"{label} needs image_run_dirs and multimodal_run_dirs")
        image = _validate_arm(
            [load_ensemble_artifact(path) for path in image_paths], f"{label} image"
        )
        multimodal = _validate_arm(
            [load_ensemble_artifact(path) for path in multimodal_paths],
            f"{label} multimodal",
        )
        if image.keys() != multimodal.keys():
            raise ValueError(f"{label} arms contain different folds")

        frames, paired_folds = [], []
        for fold in sorted(image):
            fold_df, img_probs, mm_probs = _fold_frame(image[fold], multimodal[fold])
            frames.append(fold_df)
            paired_folds.append((img_probs, mm_probs, image[fold].patient_ids))
        frame = pd.concat(frames, ignore_index=True)
        if frame["img_id"].duplicated().any():
            raise ValueError(f"{label} test images repeat across folds")

        first = image[min(image)]
        classes = first.classes.tolist()
        metrics = {
            arm: _classification_metrics(frame, arm, classes)
            for arm in ("image", "multimodal")
        }
        summary = summarize_uncertainty(frame, self.bootstrap_samples, self.seed)
        error_df = _error_detection_summary(frame)
        rc_curves, rc_aurc = _risk_coverage_analysis(frame)
        fold_df = _per_fold_summary(
            label, frame, classes, self.bootstrap_samples, self.seed
        )
        stability_df = _ensemble_size_stability(paired_folds)
        curves_df = _aggregate_training_curves(image, multimodal)
        missing_df = _missing_metadata_analysis(
            frame,
            self._resolve_metadata_path(first),
            classes,
            self.bootstrap_samples,
            self.seed,
        )

        patient_agg: dict[str, tuple[str, str]] = {
            "n_images": ("img_id", "size"),
            "image_total": ("image_total", "mean"),
            "multimodal_total": ("multimodal_total", "mean"),
            "image_aleatoric": ("image_aleatoric", "mean"),
            "multimodal_aleatoric": ("multimodal_aleatoric", "mean"),
            "image_epistemic": ("image_epistemic", "mean"),
            "multimodal_epistemic": ("multimodal_epistemic", "mean"),
            "image_variance_epistemic": ("image_variance_epistemic", "mean"),
            "multimodal_variance_epistemic": ("multimodal_variance_epistemic", "mean"),
            "image_normalized_epistemic": ("image_normalized_epistemic", "mean"),
            "multimodal_normalized_epistemic": ("multimodal_normalized_epistemic", "mean"),
            "depistemic": ("depistemic", "mean"),
            "dvariance_epistemic": ("dvariance_epistemic", "mean"),
            "dnormalized_epistemic": ("dnormalized_epistemic", "mean"),
        }
        has_ts = "dts_epistemic" in frame.columns
        if has_ts:
            for col in (
                "image_ts_total",
                "multimodal_ts_total",
                "image_ts_aleatoric",
                "multimodal_ts_aleatoric",
                "image_ts_epistemic",
                "multimodal_ts_epistemic",
                "dts_epistemic",
            ):
                patient_agg[col] = (col, "mean")
        patient = frame.groupby("patient_id").agg(**patient_agg).reset_index()

        condition_dir = self.output_dir / label
        condition_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(condition_dir / "per_image.csv", index=False)
        patient.to_csv(condition_dir / "per_patient.csv", index=False)
        summary.to_csv(condition_dir / "uncertainty_summary.csv", index=False)
        error_df.to_csv(condition_dir / "error_detection_summary.csv", index=False)
        rc_curves.to_csv(condition_dir / "risk_coverage.csv", index=False)
        rc_aurc.to_csv(condition_dir / "aurc_summary.csv", index=False)
        fold_df.to_csv(condition_dir / "per_fold_summary.csv", index=False)
        stability_df.to_csv(condition_dir / "ensemble_size_stability.csv", index=False)
        (condition_dir / "classification_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )

        plot_uncertainty(frame, label, condition_dir / "uncertainty.png")
        plot_paired_eu_scatter(frame, patient, label, condition_dir / "paired_eu_scatter.png")
        plot_correct_vs_incorrect(
            frame, error_df, label, condition_dir / "correct_vs_incorrect_eu.png"
        )
        plot_risk_coverage(rc_curves, rc_aurc, label, condition_dir / "risk_coverage.png")
        plot_classification(metrics, classes, label, condition_dir / "classification.png")
        plot_calibration(frame, metrics, label, condition_dir / "calibration.png")
        plot_fold_comparison(fold_df, label, condition_dir / "fold_comparison.png")
        plot_ensemble_size_stability(
            stability_df, label, condition_dir / "ensemble_size_stability.png"
        )
        if curves_df is not None:
            curves_df.to_csv(condition_dir / "training_curves.csv", index=False)
            plot_training_curves(curves_df, label, condition_dir / "loss_accuracy_curves.png")
            curves_df = curves_df.assign(condition=label)
        if missing_df is not None:
            missing_df.to_csv(condition_dir / "missing_metadata_summary.csv", index=False)
            plot_missing_metadata_stratification(
                missing_df, label, condition_dir / "missing_metadata_stratification.png"
            )

        overall_mi = summary[
            (summary["stratum_type"] == "overall")
            & (summary["measure"] == "entropy_epistemic")
        ].iloc[0]
        overall_var = summary[
            (summary["stratum_type"] == "overall")
            & (summary["measure"] == "variance_epistemic")
        ].iloc[0]
        eu_aurc = rc_aurc[rc_aurc["score"] == "epistemic"].set_index("arm")["aurc"].to_dict()
        eu_err = (
            error_df[error_df["score"] == "epistemic"]
            .set_index("arm")["error_auroc"]
            .to_dict()
        )
        record = {
            "condition": label,
            "architecture": str(_config_value(first, "model.image_encoder.arch")),
            "train_fraction": float(_config_value(first, "data.train_fraction") or 1.0),
            "epochs": int(_config_value(first, "train.epochs") or 0),
            "folds": ",".join(str(fold) for fold in sorted(image)),
            "n_images": len(frame),
            "n_patients": int(frame["patient_id"].nunique()),
            "mean_image_total": float(patient["image_total"].mean()),
            "mean_multimodal_total": float(patient["multimodal_total"].mean()),
            "mean_image_aleatoric": float(patient["image_aleatoric"].mean()),
            "mean_multimodal_aleatoric": float(patient["multimodal_aleatoric"].mean()),
            "mean_image_epistemic": float(patient["image_epistemic"].mean()),
            "mean_multimodal_epistemic": float(patient["multimodal_epistemic"].mean()),
            "mean_image_variance_epistemic": float(
                patient["image_variance_epistemic"].mean()
            ),
            "mean_multimodal_variance_epistemic": float(
                patient["multimodal_variance_epistemic"].mean()
            ),
            "mean_image_normalized_epistemic": float(
                patient["image_normalized_epistemic"].mean()
            ),
            "mean_multimodal_normalized_epistemic": float(
                patient["multimodal_normalized_epistemic"].mean()
            ),
            "mean_deu": float(overall_mi["mean_deu"]),
            "deu_ci_low": float(overall_mi["deu_ci_low"]),
            "deu_ci_high": float(overall_mi["deu_ci_high"]),
            "wilcoxon_p": float(overall_mi["wilcoxon_p"]),
            "mean_dvariance_epistemic": float(overall_var["mean_deu"]),
            "image_accuracy": metrics["image"]["accuracy"],
            "multimodal_accuracy": metrics["multimodal"]["accuracy"],
            "image_balanced_accuracy": metrics["image"]["balanced_accuracy"],
            "multimodal_balanced_accuracy": metrics["multimodal"]["balanced_accuracy"],
            "image_macro_f1": metrics["image"]["macro_f1"],
            "multimodal_macro_f1": metrics["multimodal"]["macro_f1"],
            "image_weighted_f1": metrics["image"]["weighted_f1"],
            "multimodal_weighted_f1": metrics["multimodal"]["weighted_f1"],
            "image_mel_recall": metrics["image"].get("mel_recall"),
            "multimodal_mel_recall": metrics["multimodal"].get("mel_recall"),
            "image_mel_specificity": metrics["image"].get("mel_specificity"),
            "multimodal_mel_specificity": metrics["multimodal"].get("mel_specificity"),
            "image_macro_ovr_auroc": metrics["image"].get("macro_ovr_auroc"),
            "multimodal_macro_ovr_auroc": metrics["multimodal"].get("macro_ovr_auroc"),
            "image_nll": metrics["image"]["negative_log_likelihood"],
            "multimodal_nll": metrics["multimodal"]["negative_log_likelihood"],
            "image_brier": metrics["image"]["multiclass_brier"],
            "multimodal_brier": metrics["multimodal"]["multiclass_brier"],
            "image_ece": metrics["image"]["ece"],
            "multimodal_ece": metrics["multimodal"]["ece"],
            "image_eu_aurc": eu_aurc.get("image"),
            "multimodal_eu_aurc": eu_aurc.get("multimodal"),
            "image_eu_error_auroc": eu_err.get("image"),
            "multimodal_eu_error_auroc": eu_err.get("multimodal"),
        }
        if has_ts:
            ts_diffs = patient["dts_epistemic"].to_numpy()
            ts_lo, ts_hi = bootstrap_mean_ci(ts_diffs, self.bootstrap_samples, self.seed)
            record.update(
                {
                    "image_mean_temp": float(frame["image_mean_temp"].mean()),
                    "multimodal_mean_temp": float(frame["multimodal_mean_temp"].mean()),
                    "image_ts_nll": float(frame["image_ts_nll"].mean()),
                    "multimodal_ts_nll": float(frame["multimodal_ts_nll"].mean()),
                    "mean_image_ts_total": float(patient["image_ts_total"].mean()),
                    "mean_multimodal_ts_total": float(
                        patient["multimodal_ts_total"].mean()
                    ),
                    "mean_image_ts_aleatoric": float(
                        patient["image_ts_aleatoric"].mean()
                    ),
                    "mean_multimodal_ts_aleatoric": float(
                        patient["multimodal_ts_aleatoric"].mean()
                    ),
                    "mean_image_ts_epistemic": float(
                        patient["image_ts_epistemic"].mean()
                    ),
                    "mean_multimodal_ts_epistemic": float(
                        patient["multimodal_ts_epistemic"].mean()
                    ),
                    "mean_ts_deu": float(ts_diffs.mean()),
                    "ts_deu_ci_low": ts_lo,
                    "ts_deu_ci_high": ts_hi,
                    "ts_wilcoxon_p": _paired_test(
                        patient["image_ts_epistemic"].to_numpy(),
                        patient["multimodal_ts_epistemic"].to_numpy(),
                    ),
                }
            )
        return record, frame, fold_df, summary, curves_df, missing_df

    def run(self) -> Path:
        conditions = self._load_conditions()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "resolved_manifest.json").write_text(
            json.dumps({"conditions": conditions}, indent=2)
        )

        records: list[dict[str, Any]] = []
        condition_frames: dict[str, pd.DataFrame] = {}
        condition_summaries: dict[str, pd.DataFrame] = {}
        condition_missing: dict[str, pd.DataFrame] = {}
        fold_tables: list[pd.DataFrame] = []
        curve_tables: list[pd.DataFrame] = []

        for label, spec in conditions.items():
            record, frame, fold_df, summary, curves_df, missing_df = (
                self._analyze_condition(label, spec)
            )
            records.append(record)
            condition_frames[label] = frame
            condition_summaries[label] = summary
            fold_tables.append(fold_df)
            if curves_df is not None:
                curve_tables.append(curves_df)
            if missing_df is not None:
                condition_missing[label] = missing_df

        comparison = _attach_budget_regime(pd.DataFrame(records))
        all_folds = pd.concat(fold_tables, ignore_index=True)
        agreement = _eu_measures_agreement(condition_frames)
        positive_control = _positive_control_summary(
            comparison, condition_frames, self.bootstrap_samples, self.seed
        )

        comparison.to_csv(self.output_dir / "condition_comparison.csv", index=False)
        (self.output_dir / "condition_comparison.json").write_text(
            comparison.to_json(orient="records", indent=2)
        )
        all_folds.to_csv(self.output_dir / "all_folds_comparison.csv", index=False)
        agreement.to_csv(self.output_dir / "eu_measures_agreement.csv", index=False)

        plot_condition_comparison(comparison, self.output_dir / "condition_comparison.png")
        plot_eu_measures_comparison(
            comparison, self.output_dir / "eu_measures_comparison.png"
        )
        plot_cross_setup_fold_eu(all_folds, self.output_dir / "cross_setup_fold_eu.png")
        if curve_tables:
            all_curves = pd.concat(curve_tables, ignore_index=True)
            all_curves.to_csv(self.output_dir / "all_training_curves.csv", index=False)
            plot_cross_setup_learning_curves(
                all_curves, self.output_dir / "cross_setup_learning_curves.png"
            )
        if comparison["train_fraction"].nunique() > 1:
            plot_train_fraction(comparison, self.output_dir / "train_fraction.png")
        if positive_control is not None:
            positive_control.to_csv(
                self.output_dir / "positive_control_summary.csv", index=False
            )

        _write_summary_tables_md(
            self.output_dir,
            comparison,
            all_folds,
            condition_summaries,
            condition_missing,
            positive_control,
            condition_frames,
        )
        return self.output_dir

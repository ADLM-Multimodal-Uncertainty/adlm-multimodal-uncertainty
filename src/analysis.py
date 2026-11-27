"""Paired comparison of an image-only and an image + metadata deep ensemble.

    uv run python -m src.analysis exp11_img:exp11_mm
    uv run python -m src.analysis exp11_img:exp11_mm exp22_img_wl:exp22_mm_wl \\
        --runs-root outputs/runs

Each argument is `<image-only name>:<multimodal name>`, the `name=` values given to
`src.train`. All folds found under `--runs-root` (`<name>-<split>-f<fold>-s<seed>/`) are
pooled, so with k-fold splits every image is scored exactly once per arm.

Recomputed from each fold's `test_member_predictions.npz` (and cross-checked against the
saved `test_metrics.json` and `test_predictions.csv`):

- accuracy, balanced accuracy, macro-F1, per-class recall, expected calibration error (ECE),
  pooled over folds and per fold, for the ensemble and as the mean of its single members;
- total, aleatoric and epistemic uncertainty per image (same definition as
  `src.uq.deep_ensemble.uncertainty`, nats), their means, and mean EU per class;
- dEU = EU(image only) - EU(image + metadata) per image. Positive means the metadata lowered
  the epistemic uncertainty. Means come with 95 % percentile intervals from a bootstrap
  that resamples patients (images of one patient are not independent), also per class
  and for the differences in balanced accuracy, macro-F1 and accuracy.

Temperature scaling: when every fold also has `val_member_predictions.npz` (written by
`src.evaluate`), one temperature per member and fold is fitted on that fold's validation
set (minimum negative log-likelihood) and applied to the member's test probabilities. All
numbers above are then reported a second time after scaling. Runs evaluated before this
file existed can be re-evaluated with `python -m src.evaluate <run_dir>` (no retraining).

Writes to `--out/<image-only>_vs_<multimodal>/`: `per_image.csv` (every number per image),
`summary.json`, `summary.md`; and `--out/report.md` with an overview of all pairs (one row
per imbalance strategy, for example) followed by each pair's summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

from src import CLASSES
from src.uq.deep_ensemble import uncertainty

ECE_BINS = 15
TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------- loading


@dataclass
class Arm:
    """One ensemble experiment, all folds pooled. Test arrays are aligned by image."""

    name: str
    folds: list[int]
    configs: dict[int, dict]
    fold: np.ndarray  # (N,)
    img_ids: np.ndarray  # (N,)
    patient_ids: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) class indices
    member_probs: np.ndarray  # (M, N, K)
    val: dict[int, tuple[np.ndarray, np.ndarray]] | None = None  # fold -> (M,Nv,K), (Nv,)

    @property
    def probs(self) -> np.ndarray:
        return self.member_probs.mean(axis=0)


def find_runs(runs_root: Path, name: str) -> dict[int, Path]:
    """Fold -> run directory of the ensemble experiment called `name`."""
    runs: dict[int, Path] = {}
    for run_dir in sorted(runs_root.glob(f"{name}-*")):
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        if cfg.get("name") != name:
            continue
        if cfg.get("ensemble") is None:
            raise ValueError(f"{run_dir} is not a deep ensemble run")
        if not (run_dir / "test_member_predictions.npz").is_file():
            raise FileNotFoundError(f"{run_dir} has no test_member_predictions.npz (unfinished?)")
        fold = int(cfg["fold"])
        if fold in runs:
            raise ValueError(f"two runs of {name} for fold {fold}: {runs[fold]} and {run_dir}")
        runs[fold] = run_dir
    if not runs:
        raise FileNotFoundError(f"no finished ensemble runs named {name!r} under {runs_root}")
    return runs


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        saved = {key: npz[key] for key in npz.files}
    if list(saved["classes"]) != CLASSES:
        raise ValueError(f"{path}: unexpected class order {list(saved['classes'])}")
    return saved


def check_saved_outputs(run_dir: Path, saved: dict, labels: np.ndarray) -> None:
    """The npz must agree with the CSV and JSON written by the pipeline."""
    probs = saved["probabilities"].mean(axis=0)
    predictions = pd.read_csv(run_dir / "test_predictions.csv", dtype={"img_id": str})
    if not np.array_equal(predictions["img_id"].to_numpy(dtype=str), saved["img_ids"]):
        raise ValueError(f"{run_dir}: image order differs between npz and test_predictions.csv")
    _, _, epistemic = uncertainty(saved["probabilities"])
    if not np.allclose(predictions["epistemic_uncertainty"], epistemic, atol=1e-9):
        raise ValueError(f"{run_dir}: epistemic uncertainty differs from test_predictions.csv")
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    ours = classification_metrics(labels, probs.argmax(axis=1))
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not np.isclose(metrics[key], ours[key], atol=1e-9):
            raise ValueError(f"{run_dir}: {key} differs from test_metrics.json")


def load_arm(runs_root: Path, name: str) -> Arm:
    runs = find_runs(runs_root, name)
    parts, configs, val = [], {}, {}
    for fold, run_dir in sorted(runs.items()):
        saved = read_npz(run_dir / "test_member_predictions.npz")
        labels = saved["labels"].astype(int)
        check_saved_outputs(run_dir, saved, labels)
        parts.append((fold, saved, labels))
        configs[fold] = yaml.safe_load((run_dir / "config.yaml").read_text())
        val_path = run_dir / "val_member_predictions.npz"
        if val_path.is_file():
            saved_val = read_npz(val_path)
            if set(saved_val["patient_ids"]) & set(saved["patient_ids"]):
                raise ValueError(f"{run_dir}: validation and test share patients")
            val[fold] = (saved_val["probabilities"], saved_val["labels"].astype(int))

    n_members = {saved["probabilities"].shape[0] for _, saved, _ in parts}
    if len(n_members) != 1:
        raise ValueError(f"{name}: folds have different ensemble sizes {sorted(n_members)}")
    arm = Arm(
        name=name,
        folds=[fold for fold, _, _ in parts],
        configs=configs,
        fold=np.concatenate([np.full(len(lab), fold) for fold, _, lab in parts]),
        img_ids=np.concatenate([saved["img_ids"] for _, saved, _ in parts]).astype(str),
        patient_ids=np.concatenate([saved["patient_ids"] for _, saved, _ in parts]).astype(str),
        labels=np.concatenate([lab for _, _, lab in parts]),
        member_probs=np.concatenate([saved["probabilities"] for _, saved, _ in parts], axis=1),
        val=val if set(val) == set(runs) else None,
    )
    if len(set(arm.img_ids)) != len(arm.img_ids):
        raise ValueError(f"{name}: an image is in the test set of more than one fold")
    return arm


def align(image_only: Arm, multimodal: Arm) -> tuple[Arm, list[str]]:
    """Reorder `multimodal` to the image order of `image_only`; return it and any warnings."""
    warnings = []
    if image_only.folds != multimodal.folds:
        raise ValueError(f"different folds: {image_only.folds} vs {multimodal.folds}")
    for fold in image_only.folds:
        a, b = image_only.configs[fold], multimodal.configs[fold]
        for key in ("split", "seed"):
            if a.get(key) != b.get(key):
                raise ValueError(f"fold {fold}: '{key}' differs between the arms")
        for key in ("ensemble", "train"):
            if a.get(key) != b.get(key):
                warnings.append(f"fold {fold}: '{key}' settings differ between the arms")
        if a["model"].get("image_encoder") != b["model"].get("image_encoder"):
            warnings.append(f"fold {fold}: image encoder settings differ between the arms")
    if a["model"].get("tabular_encoder") is not None:
        warnings.append(f"{image_only.name} has a metadata encoder: is it really image-only?")
    if b["model"].get("tabular_encoder") is None:
        warnings.append(f"{multimodal.name} has no metadata encoder: is it really multimodal?")

    order = pd.Series(np.arange(len(multimodal.img_ids)), index=multimodal.img_ids)
    if set(order.index) != set(image_only.img_ids):
        raise ValueError("the two arms were not tested on the same images")
    idx = order.loc[image_only.img_ids].to_numpy()
    for field in ("fold", "patient_ids", "labels"):
        if not np.array_equal(getattr(multimodal, field)[idx], getattr(image_only, field)):
            raise ValueError(f"'{field}' differs between the arms for the same images")
    aligned = replace(
        multimodal, fold=multimodal.fold[idx], img_ids=multimodal.img_ids[idx],
        patient_ids=multimodal.patient_ids[idx], labels=multimodal.labels[idx],
        member_probs=multimodal.member_probs[:, idx],
    )
    return aligned, warnings


# --------------------------------------------------------------------------- calibration


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log p / T): identical to dividing the logits by T."""
    logits = np.log(np.clip(probs, TINY, 1)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising the mean negative log-likelihood, searched over log T."""
    def nll(log_t: float) -> float:
        scaled = apply_temperature(probs, np.exp(log_t))
        return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], TINY, 1)).mean())

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def temperature_scaled(arm: Arm) -> tuple[Arm, dict[int, list[float]]]:
    """Per member and fold: fit on that fold's validation set, apply to its test images."""
    scaled = arm.member_probs.copy()
    temperatures: dict[int, list[float]] = {}
    for fold in arm.folds:
        val_probs, val_labels = arm.val[fold]
        mask = arm.fold == fold
        temperatures[fold] = []
        for m in range(arm.member_probs.shape[0]):
            t = fit_temperature(val_probs[m], val_labels)
            temperatures[fold].append(t)
            scaled[m, mask] = apply_temperature(arm.member_probs[m, mask], t)
    return replace(arm, member_probs=scaled), temperatures


# --------------------------------------------------------------------------- metrics


def confusion(labels: np.ndarray, preds: np.ndarray, weights: np.ndarray | None = None):
    k = len(CLASSES)
    flat = np.bincount(labels * k + preds, weights=weights, minlength=k * k)
    return flat.reshape(k, k)  # rows = truth


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Same definitions as sklearn: balanced accuracy averages recall over classes present
    in the truth; macro-F1 averages F1 over all classes, 0 where undefined."""
    true_counts, pred_counts, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    present = true_counts > 0
    recall = np.divide(tp, true_counts, out=np.zeros_like(tp, dtype=float), where=present)
    precision = np.divide(
        tp, pred_counts, out=np.zeros_like(tp, dtype=float), where=pred_counts > 0
    )
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1.mean()),
        "recall": dict(zip(CLASSES, recall.tolist(), strict=True)),
    }


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    return metrics_from_confusion(confusion(labels, preds))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS):
    """Top-label ECE with equal-width confidence bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    which = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def arm_metrics(arm: Arm) -> dict:
    """Ensemble metrics pooled and per fold, plus the mean over single members."""
    def block(member_probs: np.ndarray, labels: np.ndarray) -> dict:
        probs = member_probs.mean(axis=0)
        total, aleatoric, epistemic = uncertainty(member_probs)
        members = [classification_metrics(labels, p.argmax(axis=1)) for p in member_probs]
        return {
            **classification_metrics(labels, probs.argmax(axis=1)),
            "ece": expected_calibration_error(probs, labels),
            "mean_total_uncertainty": float(total.mean()),
            "mean_aleatoric_uncertainty": float(aleatoric.mean()),
            "mean_epistemic_uncertainty": float(epistemic.mean()),
            "mean_epistemic_per_class": {
                c: float(epistemic[labels == k].mean())
                for k, c in enumerate(CLASSES) if (labels == k).any()
            },
            "member_mean": {
                key: float(np.mean([m[key] for m in members]))
                for key in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            "n_images": int(len(labels)),
        }

    per_fold = {
        int(f): block(arm.member_probs[:, arm.fold == f], arm.labels[arm.fold == f])
        for f in arm.folds
    }
    return {"pooled": block(arm.member_probs, arm.labels), "per_fold": per_fold}


# --------------------------------------------------------------------------- bootstrap


def patient_bootstrap_weights(patient_ids: np.ndarray, n_boot: int, seed: int):
    """(n_boot, N) image weights: how often each image's patient was drawn."""
    codes, uniques = pd.factorize(patient_ids)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniques), size=(n_boot, len(uniques)))
    patient_counts = np.stack([np.bincount(d, minlength=len(uniques)) for d in draws])
    return patient_counts[:, codes].astype(float)


def interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def mean_with_ci(x: np.ndarray, patient_ids: np.ndarray, n_boot: int, seed: int) -> dict:
    w = patient_bootstrap_weights(patient_ids, n_boot, seed)
    boot = (w @ x) / w.sum(axis=1)
    return {"mean": float(x.mean()), "ci95": interval(boot), "n_images": int(len(x)),
            "n_patients": int(len(set(patient_ids)))}


def metric_difference_ci(
    labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, patient_ids: np.ndarray,
    n_boot: int, seed: int,
) -> dict:
    """b - a for balanced accuracy, macro-F1 and accuracy, with a paired patient bootstrap."""
    w = patient_bootstrap_weights(patient_ids, n_boot, seed)
    point_a = classification_metrics(labels, preds_a)
    point_b = classification_metrics(labels, preds_b)
    out = {}
    for key in ("balanced_accuracy", "macro_f1", "accuracy"):
        boot = np.array([
            metrics_from_confusion(confusion(labels, preds_b, wi))[key]
            - metrics_from_confusion(confusion(labels, preds_a, wi))[key]
            for wi in w
        ])
        out[key] = {"difference": point_b[key] - point_a[key], "ci95": interval(boot)}
    return out


# --------------------------------------------------------------------------- comparison


def paired_block(image_only: Arm, multimodal: Arm, n_boot: int, seed: int):
    """All numbers for one pair of (already aligned) arms, plus the per-image columns."""
    unc_img, unc_mm = uncertainty(image_only.member_probs), uncertainty(multimodal.member_probs)
    preds_img, preds_mm = image_only.probs.argmax(axis=1), multimodal.probs.argmax(axis=1)
    delta = unc_img[2] - unc_mm[2]
    patients, labels = image_only.patient_ids, image_only.labels

    columns = {
        "pred_image_only": [CLASSES[i] for i in preds_img],
        "pred_multimodal": [CLASSES[i] for i in preds_mm],
        "correct_image_only": preds_img == labels,
        "correct_multimodal": preds_mm == labels,
    }
    for tag, (total, aleatoric, epistemic), arm in (
        ("image_only", unc_img, image_only), ("multimodal", unc_mm, multimodal)
    ):
        columns[f"total_{tag}"] = total
        columns[f"aleatoric_{tag}"] = aleatoric
        columns[f"epistemic_{tag}"] = epistemic
        for k, c in enumerate(CLASSES):
            columns[f"p_{c}_{tag}"] = arm.probs[:, k]
    columns["delta_epistemic"] = delta

    per_class = {
        c: mean_with_ci(delta[labels == k], patients[labels == k], n_boot, seed)
        for k, c in enumerate(CLASSES) if (labels == k).any()
    }
    mean_img = unc_img[2].mean()
    block = {
        "metrics": {"image_only": arm_metrics(image_only), "multimodal": arm_metrics(multimodal)},
        "delta_epistemic": {
            **mean_with_ci(delta, patients, n_boot, seed),
            "relative_to_image_only": float(delta.mean() / mean_img) if mean_img > 0 else None,
            "share_images_lowered": float((delta > 0).mean()),
            "per_class": per_class,
            "per_fold": {
                int(f): float(delta[image_only.fold == f].mean()) for f in image_only.folds
            },
        },
        "multimodal_minus_image_only": metric_difference_ci(
            labels, preds_img, preds_mm, patients, n_boot, seed
        ),
    }
    return block, columns


def compare(image_only: Arm, multimodal: Arm, n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    multimodal, warnings = align(image_only, multimodal)
    raw, columns = paired_block(image_only, multimodal, n_boot, seed)
    per_image = pd.DataFrame({
        "img_id": image_only.img_ids,
        "patient_id": image_only.patient_ids,
        "fold": image_only.fold,
        "diagnostic": [CLASSES[i] for i in image_only.labels],
        **columns,
    })

    scaled, temperatures = None, None
    if image_only.val is not None and multimodal.val is not None:
        img_ts, t_img = temperature_scaled(image_only)
        mm_ts, t_mm = temperature_scaled(multimodal)
        scaled, columns_ts = paired_block(img_ts, mm_ts, n_boot, seed)
        temperatures = {"image_only": t_img, "multimodal": t_mm}
        for key, values in columns_ts.items():
            per_image[f"{key}_ts"] = values
    else:
        warnings.append(
            "no val_member_predictions.npz for every fold of both arms: temperature scaling "
            "skipped (re-run `python -m src.evaluate <run_dir>` to create them)"
        )

    summary = {
        "image_only": image_only.name,
        "multimodal": multimodal.name,
        "folds": image_only.folds,
        "n_members": int(image_only.member_probs.shape[0]),
        "bootstrap": {"n_boot": n_boot, "seed": seed, "unit": "patient"},
        "warnings": warnings,
        "uncalibrated": raw,
        "temperature_scaled": scaled,
        "temperatures": temperatures,
    }
    return summary, per_image


# --------------------------------------------------------------------------- report


def fmt_ci(d: dict, key: str = "mean", digits: int = 4) -> str:
    low, high = d["ci95"]
    return f"{d[key]:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def conclusion(d: dict) -> str:
    low, high = d["ci95"]
    if low > 0:
        return "metadata lowers EU"
    if high < 0:
        return "metadata raises EU"
    return "no clear change"


def block_markdown(block: dict, title: str) -> list[str]:
    img, mm = block["metrics"]["image_only"], block["metrics"]["multimodal"]
    diff = block["multimodal_minus_image_only"]
    d = block["delta_epistemic"]

    def fold_spread(arm: dict, key: str) -> str:
        values = [f[key] for f in arm["per_fold"].values()]
        return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

    rows = [("Balanced accuracy", "balanced_accuracy"), ("Macro-F1", "macro_f1"),
            ("Accuracy", "accuracy")]
    lines = [f"#### {title}", "",
             "| Metric | Image only | Image + metadata | Difference (95 % CI) |",
             "|---|---|---|---|"]
    for label, key in rows:
        lines.append(f"| {label} | {img['pooled'][key]:.3f} | {mm['pooled'][key]:.3f} | "
                     f"{fmt_ci(diff[key], 'difference', 3)} |")
    for label, key in rows[:2]:
        lines.append(f"| {label}, per fold (mean ± sd) | {fold_spread(img, key)} | "
                     f"{fold_spread(mm, key)} | |")
        lines.append(f"| {label}, mean single member | {img['pooled']['member_mean'][key]:.3f} "
                     f"| {mm['pooled']['member_mean'][key]:.3f} | |")
    for c in CLASSES:
        lines.append(f"| Recall {c} | {img['pooled']['recall'][c]:.3f} | "
                     f"{mm['pooled']['recall'][c]:.3f} | |")
    lines.append(f"| ECE | {img['pooled']['ece']:.3f} | {mm['pooled']['ece']:.3f} | |")
    for label, key in (("total", "mean_total_uncertainty"),
                       ("aleatoric", "mean_aleatoric_uncertainty"),
                       ("epistemic", "mean_epistemic_uncertainty")):
        lines.append(f"| Mean {label} uncertainty (nats) | {img['pooled'][key]:.4f} | "
                     f"{mm['pooled'][key]:.4f} | |")
    for c in img["pooled"]["mean_epistemic_per_class"]:
        lines.append(f"| Mean EU {c} | {img['pooled']['mean_epistemic_per_class'][c]:.4f} | "
                     f"{mm['pooled']['mean_epistemic_per_class'][c]:.4f} | |")
    lines += [
        "",
        "ΔEU = EU(image only) − EU(image + metadata), positive: metadata lowers EU.",
        "",
        "| | Mean ΔEU (95 % CI) | Images | Patients |",
        "|---|---|---|---|",
        f"| All | {fmt_ci(d)} | {d['n_images']} | {d['n_patients']} |",
    ]
    for c, v in d["per_class"].items():
        lines.append(f"| {c} | {fmt_ci(v)} | {v['n_images']} | {v['n_patients']} |")
    rel = d["relative_to_image_only"]
    lines += [
        "",
        (f"Relative to image-only EU: {rel:+.1%}. " if rel is not None else "")
        + f"Images with lower EU with metadata: {d['share_images_lowered']:.1%}. "
        + "Mean ΔEU per fold: "
        + ", ".join(f"f{f} {v:+.4f}" for f, v in d["per_fold"].items()) + ".",
        "",
    ]
    return lines


def summary_markdown(s: dict) -> str:
    raw = s["uncalibrated"]["delta_epistemic"]
    lines = [
        f"### {s['image_only']} (image only) vs {s['multimodal']} (image + metadata)",
        "",
        f"Folds {s['folds']}, {s['n_members']} members per ensemble, {raw['n_images']} test "
        f"images from {raw['n_patients']} patients (all folds pooled). 95 % intervals: "
        f"{s['bootstrap']['n_boot']} patient-level bootstrap resamples.",
        "",
    ]
    lines += [f"> **Warning:** {w}" for w in s["warnings"]]
    if s["warnings"]:
        lines.append("")
    lines += block_markdown(s["uncalibrated"], "Uncalibrated")
    if s["temperature_scaled"] is not None:
        temps = s["temperatures"]
        lines.append(
            "Temperatures fitted on validation (one per member and fold): image only "
            + ", ".join(f"f{f} {np.round(t, 2).tolist()}" for f, t in temps["image_only"].items())
            + "; image + metadata "
            + ", ".join(f"f{f} {np.round(t, 2).tolist()}" for f, t in temps["multimodal"].items())
            + "."
        )
        lines.append("")
        lines += block_markdown(s["temperature_scaled"], "After temperature scaling")
    return "\n".join(lines)


def ece_change(raw: dict, ts: dict | None, arm: str) -> str:
    before = raw["metrics"][arm]["pooled"]["ece"]
    if ts is None:
        return f"{before:.3f}"
    return f"{before:.3f} → {ts['metrics'][arm]['pooled']['ece']:.3f}"


def overview_markdown(summaries: list[dict]) -> str:
    """One row per pair: the quantities used to choose an imbalance strategy."""
    lines = [
        "## Overview",
        "",
        "Image only / image + metadata. ECE: before → after temperature scaling.",
        "",
        "| Image only : multimodal | Bal. acc. | Macro-F1 | MEL recall | ECE image only | "
        "ECE image + meta | ΔEU (95 % CI) | ΔEU after TS (95 % CI) | Conclusion (TS) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        raw, ts = s["uncalibrated"], s["temperature_scaled"]
        img, mm = raw["metrics"]["image_only"]["pooled"], raw["metrics"]["multimodal"]["pooled"]

        d_raw = raw["delta_epistemic"]
        d_ts = ts["delta_epistemic"] if ts is not None else None
        lines.append(
            f"| {s['image_only']} : {s['multimodal']} "
            f"| {img['balanced_accuracy']:.3f} / {mm['balanced_accuracy']:.3f} "
            f"| {img['macro_f1']:.3f} / {mm['macro_f1']:.3f} "
            f"| {img['recall']['MEL']:.3f} / {mm['recall']['MEL']:.3f} "
            f"| {ece_change(raw, ts, 'image_only')} | {ece_change(raw, ts, 'multimodal')} "
            f"| {fmt_ci(d_raw)} "
            f"| {fmt_ci(d_ts) if d_ts else '–'} "
            f"| {conclusion(d_ts) if d_ts else conclusion(d_raw) + ' (uncalibrated)'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", help="<image-only name>:<multimodal name>")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summaries, texts = [], []
    for pair in args.pairs:
        if pair.count(":") != 1:
            parser.error(f"expected <image-only name>:<multimodal name>, got {pair!r}")
        img_name, mm_name = pair.split(":")
        summary, per_image = compare(
            load_arm(args.runs_root, img_name), load_arm(args.runs_root, mm_name),
            args.n_boot, args.seed,
        )
        out_dir = args.out / f"{img_name}_vs_{mm_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        per_image.to_csv(out_dir / "per_image.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        text = summary_markdown(summary)
        (out_dir / "summary.md").write_text(text)
        summaries.append(summary)
        texts.append(text)
    report = "\n".join(["# ΔEU analysis", "", overview_markdown(summaries), *texts])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report)
    print(report)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

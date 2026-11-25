"""PAD-UFES-20 loading and the train/val/test datasets of one fold."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from src import CLASSES
from src.data.metadata_encoding import clean_features, make_preprocessor
from src.data.splits import Split
from src.data.transforms import image_transform

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str, "img_id": str})
    df.columns = df.columns.str.strip()
    df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()
    unknown = set(df["diagnostic"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected diagnostic labels: {sorted(unknown)}")
    if df["img_id"].duplicated().any():
        raise ValueError("duplicate img_id rows in metadata")
    return df


def build_image_index(image_dir: str | Path) -> dict[str, Path]:
    """Map image stem -> path, searching `image_dir` recursively."""
    index: dict[str, Path] = {}
    for path in Path(image_dir).rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                raise ValueError(f"duplicate image {path.stem} under {image_dir}")
            index[path.stem] = path
    if not index:
        raise FileNotFoundError(f"no images found under {image_dir}")
    return index


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        image_index: dict[str, Path],
        transform,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.features = np.asarray(features, dtype=np.float32)
        self.paths = [image_index.get(Path(i).stem) for i in self.df["img_id"]]
        missing = [i for i, p in zip(self.df["img_id"], self.paths, strict=True) if p is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:5]}")
        self.transform = transform
        self.labels = self.df["diagnostic"].map(CLASSES.index).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        with Image.open(self.paths[i]) as img:
            image = self.transform(ImageOps.exif_transpose(img).convert("RGB"))
        return {
            "image": image,
            "meta": torch.from_numpy(self.features[i]),
            "label": torch.tensor(self.labels[i]),
            "img_id": self.df.at[i, "img_id"],
            "patient_id": self.df.at[i, "patient_id"],
        }


@dataclass
class FoldData:
    train: SkinLesionDataset
    val: SkinLesionDataset
    test: SkinLesionDataset
    n_meta_features: int


def subsample_training_patients(
    df: pd.DataFrame, fraction: float, seed: int
) -> pd.DataFrame:
    """Select nested, diagnosis-stratified patient subsets for the size control."""
    allowed = (0.25, 0.5, 1.0)
    match = next((value for value in allowed if np.isclose(fraction, value)), None)
    if match is None:
        raise ValueError(f"train_fraction must be one of {allowed}, got {fraction}")
    if match == 1.0:
        return df.copy()

    def patient_label(values: pd.Series) -> str:
        counts = values.value_counts()
        return sorted(counts[counts == counts.max()].index)[0]

    patients = (
        df.groupby("patient_id", sort=True)["diagnostic"]
        .agg(patient_label)
        .rename("diagnostic")
        .reset_index()
    )
    class_patient_counts = patients["diagnostic"].value_counts()
    if (class_patient_counts < 4).any():
        sparse = class_patient_counts[class_patient_counts < 4].to_dict()
        raise ValueError(f"need at least four training patients per class, got {sparse}")

    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    patient_folds = [
        test
        for _, test in splitter.split(
            np.zeros(len(patients), dtype=np.uint8), patients["diagnostic"]
        )
    ]
    fold_count = int(match * 4)
    selected_indices = np.concatenate(patient_folds[:fold_count])
    selected_patients = set(patients.iloc[selected_indices]["patient_id"])
    selected = df[df["patient_id"].isin(selected_patients)].copy()

    missing_classes = set(df["diagnostic"]) - set(selected["diagnostic"])
    if missing_classes:
        raise ValueError(
            f"train_fraction={match} omitted classes: {sorted(missing_classes)}"
        )
    return selected


def _resolve_runtime_data_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Use a node-local mirror (`ADLM_LOCAL_DATA_ROOT`) when staged on cluster scratch,
    keeping canonical NFS paths in saved Hydra `config.yaml`."""
    metadata_csv = Path(cfg.metadata_csv)
    image_dir = Path(cfg.image_dir)
    local_root = os.environ.get("ADLM_LOCAL_DATA_ROOT")
    if local_root:
        local_path = Path(local_root)
        local_meta = local_path / "metadata.csv"
        local_images = local_path / "images"
        if local_meta.is_file() and local_images.is_dir():
            return local_meta, local_images
    return metadata_csv, image_dir


def build_fold(cfg: DictConfig, split: Split, fold: int) -> FoldData:
    """Partition with `split`, fit the metadata preprocessor on the training rows only, and
    build the three datasets. Deterministic, so evaluation can rebuild it from the config."""
    metadata_csv, image_dir = _resolve_runtime_data_paths(cfg)
    parts = split.partition(load_metadata(metadata_csv), fold)
    if cfg.subset:
        parts = {name: part.head(int(cfg.subset)) for name, part in parts.items()}
    elif not np.isclose(float(cfg.train_fraction), 1.0):
        parts["train"] = subsample_training_patients(
            parts["train"], float(cfg.train_fraction), int(cfg.train_subset_seed)
        )

    numeric, categorical = list(cfg.numeric_features), list(cfg.categorical_features)
    preprocessor = make_preprocessor(numeric, categorical)
    preprocessor.fit(clean_features(parts["train"], numeric, categorical))
    features = {
        name: preprocessor.transform(clean_features(part, numeric, categorical))
        for name, part in parts.items()
    }

    index = build_image_index(image_dir)
    datasets = {
        name: SkinLesionDataset(
            parts[name], features[name], index, image_transform(cfg.image_size, name == "train")
        )
        for name in parts
    }
    return FoldData(**datasets, n_meta_features=features["train"].shape[1])

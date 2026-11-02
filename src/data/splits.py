"""Patient-disjoint train/val/test partitions, selected by `configs/split/<name>.yaml`.

Every split exposes `partition(df, fold) -> {"train", "val", "test"}`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class Split(Protocol):
    def partition(self, df: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]: ...


def check_patient_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, part in parts.items():
        for patient in set(part["patient_id"]):
            if patient in seen:
                raise ValueError(f"patient {patient} is in both {seen[patient]} and {name}")
            seen[patient] = name


class PatientHoldoutSplit:
    """Single holdout split by patient, retried with seed + attempt until every class
    occurs in every partition. `fold` is ignored."""

    def __init__(
        self, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42, attempts: int = 100
    ) -> None:
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.attempts = attempts

    def partition(self, df: pd.DataFrame, fold: int = 0) -> dict[str, pd.DataFrame]:
        required = set(df["diagnostic"])
        for attempt in range(self.attempts):
            seed = self.seed + attempt
            outer = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=seed)
            train_val_idx, test_idx = next(outer.split(df, groups=df["patient_id"]))
            train_val = df.iloc[train_val_idx]
            inner = GroupShuffleSplit(
                n_splits=1, test_size=self.val_size / (1 - self.test_size), random_state=seed
            )
            train_idx, val_idx = next(inner.split(train_val, groups=train_val["patient_id"]))
            parts = {
                "train": train_val.iloc[train_idx],
                "val": train_val.iloc[val_idx],
                "test": df.iloc[test_idx],
            }
            if all(set(part["diagnostic"]) == required for part in parts.values()):
                check_patient_disjoint(parts)
                return parts
        raise RuntimeError(
            f"no patient-disjoint split with every class in every partition after "
            f"{self.attempts} attempts"
        )


class PatientKFoldSplit:
    """Fold assignment read from a CSV (img_id, patient_id, fold).

    Test is `fold`, validation is (fold + 1) % n_folds, training is the rest.
    """

    def __init__(self, csv: str | Path) -> None:
        self.csv = Path(csv)

    def partition(self, df: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]:
        assignment = pd.read_csv(self.csv, usecols=["img_id", "fold"], dtype={"img_id": str})
        merged = df.merge(assignment, on="img_id", how="left", validate="one_to_one")
        if merged["fold"].isna().any():
            raise ValueError(f"{int(merged['fold'].isna().sum())} rows missing from {self.csv}")
        n_folds = int(merged["fold"].max()) + 1
        val_fold = (fold + 1) % n_folds
        parts = {
            "train": merged[~merged["fold"].isin([fold, val_fold])],
            "val": merged[merged["fold"] == val_fold],
            "test": merged[merged["fold"] == fold],
        }
        check_patient_disjoint(parts)
        return parts

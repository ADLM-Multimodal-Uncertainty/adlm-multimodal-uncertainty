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

    def __init__(self, csv: str | Path, n_folds: int = 5) -> None:
        if n_folds < 3:
            raise ValueError("n_folds must be at least 3 for train/val/test partitions")
        self.csv = Path(csv)
        self.n_folds = n_folds

    def partition(self, df: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]:
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold must be in [0, {self.n_folds - 1}], got {fold}")
        assignment = pd.read_csv(
            self.csv, usecols=["img_id", "patient_id", "fold"],
            dtype={"img_id": str, "patient_id": str},
        )
        if assignment.isna().any().any():
            raise ValueError("fold assignment contains missing values")
        if assignment["img_id"].duplicated().any():
            raise ValueError("fold assignment contains duplicate img_id rows")
        if not assignment["fold"].isin(range(self.n_folds)).all():
            raise ValueError("fold assignment must contain integer fold IDs in range")
        if set(assignment["fold"]) != set(range(self.n_folds)):
            raise ValueError("fold assignment must contain every configured fold")
        if (assignment.groupby("patient_id")["fold"].nunique() > 1).any():
            raise ValueError("fold assignment places a patient in multiple folds")
        merged = df.merge(
            assignment, on="img_id", how="left", validate="one_to_one",
            suffixes=("", "_assigned"),
        )
        if merged["fold"].isna().any():
            raise ValueError(f"{int(merged['fold'].isna().sum())} rows missing from {self.csv}")
        if not merged["patient_id"].eq(merged["patient_id_assigned"]).all():
            raise ValueError("patient_id in metadata disagrees with fold assignment")
        merged = merged.drop(columns="patient_id_assigned")
        val_fold = (fold + 1) % self.n_folds
        parts = {
            "train": merged[~merged["fold"].isin([fold, val_fold])],
            "val": merged[merged["fold"] == val_fold],
            "test": merged[merged["fold"] == fold],
        }
        if any(part.empty for part in parts.values()):
            raise ValueError("metadata must contain patients in train, val and test partitions")
        check_patient_disjoint(parts)
        return parts

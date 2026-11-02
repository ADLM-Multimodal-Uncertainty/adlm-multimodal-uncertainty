import pandas as pd
import pytest

from src.data.splits import PatientHoldoutSplit, PatientKFoldSplit, check_patient_disjoint

from .conftest import REPO, synthetic_metadata

KFOLD_CSV = REPO / "data" / "splits" / "pad_ufes20_5fold_patient.csv"


def _assert_valid(parts: dict[str, pd.DataFrame], df: pd.DataFrame) -> None:
    check_patient_disjoint(parts)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert set().union(*(set(p["img_id"]) for p in parts.values())) == set(df["img_id"])


def test_holdout_is_patient_disjoint_with_every_class_everywhere():
    df = synthetic_metadata()
    parts = PatientHoldoutSplit(seed=42).partition(df)
    _assert_valid(parts, df)
    for part in parts.values():
        assert set(part["diagnostic"]) == set(df["diagnostic"])


def test_holdout_is_deterministic_and_seed_dependent():
    df = synthetic_metadata()
    a = PatientHoldoutSplit(seed=42).partition(df)
    b = PatientHoldoutSplit(seed=42).partition(df)
    c = PatientHoldoutSplit(seed=7).partition(df)
    assert all(a[k]["img_id"].tolist() == b[k]["img_id"].tolist() for k in a)
    assert a["test"]["img_id"].tolist() != c["test"]["img_id"].tolist()


@pytest.mark.parametrize("fold", range(5))
def test_kfold_uses_fold_as_test_and_next_fold_as_val(fold):
    df = pd.read_csv(KFOLD_CSV, dtype={"img_id": str}).drop(columns="fold")
    parts = PatientKFoldSplit(KFOLD_CSV).partition(df, fold)
    _assert_valid(parts, df)
    assert set(parts["test"]["fold"]) == {fold}
    assert set(parts["val"]["fold"]) == {(fold + 1) % 5}


def test_kfold_rejects_rows_missing_from_the_assignment():
    df = synthetic_metadata()
    with pytest.raises(ValueError, match="missing"):
        PatientKFoldSplit(KFOLD_CSV).partition(df, 0)


def test_disjointness_check_catches_shared_patients():
    shared = pd.DataFrame({"patient_id": ["P1"]})
    with pytest.raises(ValueError, match="P1"):
        check_patient_disjoint({"train": shared, "test": shared})

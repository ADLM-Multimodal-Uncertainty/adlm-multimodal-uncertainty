import pandas as pd
import pytest

from src.data.dataset import subsample_training_patients
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


def test_every_patient_is_tested_once_across_five_folds():
    df = pd.read_csv(KFOLD_CSV, dtype={"patient_id": str, "img_id": str}).drop(columns="fold")
    tests = [PatientKFoldSplit(KFOLD_CSV).partition(df, f)["test"] for f in range(5)]
    combined = pd.concat(tests)
    assert len(combined) == len(df)
    assert not combined["img_id"].duplicated().any()
    assert combined.groupby("patient_id")["fold"].nunique().eq(1).all()
    assert all(set(part["diagnostic"]) == set(df["diagnostic"]) for part in tests)


@pytest.mark.parametrize("fold", [-1, 5])
def test_kfold_rejects_invalid_requested_fold(fold):
    with pytest.raises(ValueError, match="fold must be"):
        PatientKFoldSplit(KFOLD_CSV).partition(synthetic_metadata(), fold)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fractional", "integer fold"),
        ("missing_fold", "every configured fold"),
        ("duplicate", "duplicate img_id"),
        ("shared_patient", "multiple folds"),
        ("wrong_patient", "disagrees"),
        ("null", "missing values"),
    ],
)
def test_kfold_rejects_invalid_assignments(tmp_path, case, message):
    df = pd.DataFrame({"img_id": list("abcdef"), "patient_id": list("ABCDEF")})
    assignment = df.assign(fold=[0, 1, 2, 3, 4, 0])
    if case == "fractional":
        assignment["fold"] = assignment["fold"].astype(float)
        assignment.loc[0, "fold"] = 0.5
    elif case == "missing_fold":
        assignment.loc[4, "fold"] = 0
    elif case == "duplicate":
        assignment = pd.concat([assignment, assignment.iloc[[0]]])
    elif case == "shared_patient":
        assignment.loc[1, "patient_id"] = "A"
    elif case == "wrong_patient":
        assignment.loc[0, "patient_id"] = "Z"
    else:
        assignment.loc[0, "patient_id"] = None
    path = tmp_path / "folds.csv"
    assignment.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        PatientKFoldSplit(path).partition(df, 0)


@pytest.mark.parametrize("n_folds", [0, 1, 2.5, True])
def test_kfold_requires_at_least_two_integer_folds(n_folds):
    with pytest.raises(ValueError, match="at least 2"):
        PatientKFoldSplit(n_folds=n_folds)


@pytest.mark.parametrize("n_folds", [2, 3, 4])
def test_generated_folds(n_folds):
    df = synthetic_metadata()
    splitter = PatientKFoldSplit(n_folds=n_folds)
    tested = []
    for fold in range(n_folds):
        parts = splitter.partition(df, fold)
        repeated = splitter.partition(df.sample(frac=1, random_state=8), fold)
        _assert_valid(parts, df)
        for name, part in parts.items():
            assert not part.empty
            assert set(part.img_id) == set(repeated[name].img_id)
        tested.append(parts["test"])
    combined = pd.concat(tested)
    assert len(combined) == len(df)
    assert not combined.img_id.duplicated().any()


def test_kfold_rejects_metadata_without_all_partitions():
    df = pd.read_csv(KFOLD_CSV, dtype={"patient_id": str, "img_id": str})
    df = df[df["fold"] == 0].drop(columns="fold")
    with pytest.raises(ValueError, match="train, val and test"):
        PatientKFoldSplit(KFOLD_CSV).partition(df, 0)


def test_default_five_fold_assignment_is_preserved():
    df = pd.read_csv(KFOLD_CSV, dtype={"img_id": str}).drop(columns="fold")
    default = PatientKFoldSplit().partition(df, 0)
    explicit = PatientKFoldSplit(KFOLD_CSV).partition(df, 0)
    assert all(default[k].equals(explicit[k]) for k in default)


def test_two_fold_csv_uses_inner_patient_validation(tmp_path):
    df = synthetic_metadata()
    assignment = df[["img_id", "patient_id"]].copy()
    assignment["fold"] = assignment["patient_id"].str[-2:].astype(int) % 2
    path = tmp_path / "two_folds.csv"
    assignment.to_csv(path, index=False)
    parts = PatientKFoldSplit(path, n_folds=2).partition(df, 0)
    _assert_valid(parts, df)
    assert all(not p.empty for p in parts.values())
    assert set(parts["test"]["fold"]) == {0}
    assert set(parts["train"]["fold"]) == set(parts["val"]["fold"]) == {1}


def test_training_patient_subsets_are_deterministic_stratified_and_nested():
    df = synthetic_metadata()
    quarter = subsample_training_patients(df, 0.25, seed=42)
    repeated = subsample_training_patients(df.sample(frac=1), 0.25, seed=42)
    half = subsample_training_patients(df, 0.5, seed=42)
    full = subsample_training_patients(df, 1.0, seed=42)

    quarter_patients = set(quarter["patient_id"])
    half_patients = set(half["patient_id"])
    full_patients = set(full["patient_id"])
    assert quarter_patients == set(repeated["patient_id"])
    assert quarter_patients < half_patients < full_patients
    assert len(quarter_patients) == 30
    assert len(half_patients) == 60
    assert all(set(part["diagnostic"]) == set(df["diagnostic"]) for part in (quarter, half))


def test_training_patient_subset_rejects_unsupported_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        subsample_training_patients(synthetic_metadata(), 0.3, seed=42)

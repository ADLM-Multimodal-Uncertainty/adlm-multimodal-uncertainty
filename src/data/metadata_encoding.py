"""Clinical metadata -> float matrix. The preprocessor is fitted on training rows only."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING = "Missing"


def clean_features(
    df: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]
) -> pd.DataFrame:
    """Numeric columns to float (NaN if missing), categorical columns to strings."""
    x = df.loc[:, [*numeric, *categorical]].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in categorical:
        x[col] = x[col].where(x[col].notna(), MISSING).astype(str)
    return x


def make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    """Median imputation + standardisation for numeric, one-hot for categorical columns.

    Categories unseen during fitting encode as all zeros.
    """
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, list(numeric)), ("cat", categorical_pipe, list(categorical))],
        sparse_threshold=0,
    )

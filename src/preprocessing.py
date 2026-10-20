"""Cleaning, train/test split and the shared preprocessing ColumnTransformer."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                     TARGET_CLF, TEST_SIZE)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates. Imputation happens later, inside the pipeline."""
    out = df.drop_duplicates().reset_index(drop=True)
    return out, {"rows_before": int(len(df)),
                 "duplicates_removed": int(len(df) - len(out)),
                 "rows_after": int(len(out))}


def split_indices(df: pd.DataFrame, seed: int):
    """Fixed, stratified train/test split with random_state=LLLL."""
    return train_test_split(df.index.to_numpy(), test_size=TEST_SIZE,
                            random_state=seed, stratify=df[TARGET_CLF])


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, one-hot season, pass binary flag through."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])

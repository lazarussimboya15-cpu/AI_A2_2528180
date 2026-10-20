"""Seed, schema, range, missing, duplicate and leakage checks."""
from __future__ import annotations

import re

import pandas as pd

from .config import (ALL_COLUMNS, FEATURES, MIN_YIELD, RANGES, SEASONS,
                     TARGET_CLF, TARGET_REG)


class ValidationError(ValueError):
    """Raised when data or arguments violate the assignment rules."""


def student_digits(student_id: str) -> str:
    """Registration digits found in the part of the ID before '@'."""
    digits = "".join(re.findall(r"\d+", str(student_id).split("@")[0]))
    if len(digits) < 4:
        raise ValidationError(f"Student ID '{student_id}' has fewer than 4 digits.")
    return digits


def validate_seed(student_id: str, seed: int) -> str:
    """Reject a seed that is not LLLL; return the verification code."""
    last4 = student_digits(student_id)[-4:]
    if int(last4) != int(seed):
        raise ValidationError(
            f"Seed {seed} does not match the last four digits ({last4}) of '{student_id}'.")
    return f"AI-A2-{last4}"


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def validate_ranges(df: pd.DataFrame) -> None:
    problems = []
    for col, (lo, hi) in RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not df["season"].dropna().isin(SEASONS).all():
        problems.append("season: values other than A/B")
    if not df["cooperative_support"].isin([0, 1]).all():
        problems.append("cooperative_support: values other than 0/1")
    if (df[TARGET_REG] < MIN_YIELD).any():
        problems.append(f"{TARGET_REG}: values below {MIN_YIELD}")
    if not df[TARGET_CLF].isin([0, 1]).all():
        problems.append("high_risk: values other than 0/1")
    if df[[TARGET_REG, TARGET_CLF]].isna().any().any():
        problems.append("targets contain missing values")
    if problems:
        raise ValidationError("Range check failed: " + "; ".join(problems))


def check_no_leakage(feature_columns: list[str]) -> None:
    """Targets and anything derived from them must never be model inputs."""
    forbidden = {TARGET_REG, TARGET_CLF, "yield_per_ha", "yield_per_hectare"}
    leaked = forbidden.intersection(feature_columns)
    if leaked:
        raise ValidationError(f"Target leakage: {sorted(leaked)} used as features.")


def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing-value and duplicate evidence on the raw data."""
    return {
        "rows_raw": int(len(df)),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "feature_columns_checked_for_leakage": FEATURES,
    }

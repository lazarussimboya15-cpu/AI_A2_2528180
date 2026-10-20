"""Shared inference: strict input validation + prediction from saved artifacts."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import DISCLAIMER, FEATURES, RANGES, ROOT, SEASONS

MODELS_DIR = ROOT / "models"


class InputValidationError(ValueError):
    """Invalid prediction input; ``errors`` holds one message per problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(record) -> dict:
    """Check fields, types and ranges; return a clean record or raise."""
    if not isinstance(record, dict):
        raise InputValidationError(["Input must be a single JSON object."])
    errors = []
    missing = [f for f in FEATURES if f not in record]
    extra = [k for k in record if k not in FEATURES]
    if missing:
        errors.append(f"Missing required field(s): {missing}")
    if extra:
        errors.append(f"Unknown field(s): {extra}")
    clean = {}
    for f in FEATURES:
        if f not in record:
            continue
        v = record[f]
        if f in RANGES:
            lo, hi = RANGES[f]
            if not _is_number(v):
                errors.append(f"{f} must be a finite number, got {v!r}")
            elif not lo <= v <= hi:
                errors.append(f"{f}={v} outside allowed range [{lo}, {hi}]")
            else:
                clean[f] = float(v)
        elif f == "season":
            if v not in SEASONS:
                errors.append(f"season must be 'A' or 'B', got {v!r}")
            else:
                clean[f] = v
        elif f == "cooperative_support":
            if isinstance(v, bool) or v not in (0, 1) or not isinstance(v, (int, float)):
                errors.append(f"cooperative_support must be 0 or 1, got {v!r}")
            else:
                clean[f] = int(v)
    if errors:
        raise InputValidationError(errors)
    return clean


@lru_cache(maxsize=4)
def load_models(models_dir: str = str(MODELS_DIR)) -> dict:
    d = Path(models_dir)
    needed = ["regressor.joblib", "classifier.joblib", "clusterer.joblib", "metadata.json"]
    absent = [n for n in needed if not (d / n).exists()]
    if absent:
        raise FileNotFoundError(f"Missing model files {absent}. Run: python run_all.py ...")
    return {"regressor": joblib.load(d / "regressor.joblib"),
            "classifier": joblib.load(d / "classifier.joblib"),
            "clusterer": joblib.load(d / "clusterer.joblib"),
            "metadata": json.loads((d / "metadata.json").read_text(encoding="utf-8"))}


def predict_record(record: dict, models_dir: Path | str = MODELS_DIR) -> dict:
    """Validate one farm record and return yield, risk, probability and cluster."""
    clean = validate_record(record)
    m = load_models(str(models_dir))
    X = pd.DataFrame([clean])[FEATURES]
    risk = int(m["classifier"].predict(X)[0])
    prob = float(m["classifier"].predict_proba(X)[0, 1])
    cluster = int(m["clusterer"].predict(X)[0])
    meta = m["metadata"]
    return {
        "verification_code": meta["verification_code"],
        "yield_estimate_tons": round(float(m["regressor"].predict(X)[0]), 4),
        "high_risk_prediction": risk,
        "risk_label": "high production risk" if risk else "lower production risk",
        "high_risk_probability": round(prob, 4),
        "cluster": cluster,
        "cluster_profile": meta["cluster_profiles"].get(str(cluster), ""),
        "regression_model": meta["regression_model"],
        "classification_model": meta["classification_model"],
        "disclaimer": DISCLAIMER,
    }

"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import DISCLAIMER, RANGES
from src.inference import InputValidationError, load_models, predict_record

app = FastAPI(title="Musanze Harvest Intelligence Service",
              description=DISCLAIMER, version="1.0.0")


def _rng(name: str) -> dict:
    lo, hi = RANGES[name]
    return {"ge": lo, "le": hi, "allow_inf_nan": False}


class FarmRecord(BaseModel):
    """One farm record. Unknown fields are rejected; ranges match the training data."""
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(..., examples=[1000.0], **_rng("rainfall_mm"))
    avg_temp_c: float = Field(..., examples=[18.0], **_rng("avg_temp_c"))
    soil_moisture_pct: float = Field(..., examples=[55.0], **_rng("soil_moisture_pct"))
    fertilizer_kg: float = Field(..., examples=[120.0], **_rng("fertilizer_kg"))
    field_size_ha: float = Field(..., examples=[1.5], **_rng("field_size_ha"))
    season: Literal["A", "B"] = Field(..., examples=["A"])
    cooperative_support: Literal[0, 1] = Field(..., examples=[1])


def _models():
    try:
        return load_models()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health() -> dict:
    """Liveness check; reports whether the saved models can be loaded."""
    try:
        meta = load_models()["metadata"]
        return {"status": "ok", "models_loaded": True,
                "verification_code": meta["verification_code"]}
    except FileNotFoundError:
        return {"status": "degraded", "models_loaded": False, "verification_code": None}


@app.get("/model-info")
def model_info() -> dict:
    """Model names, test metrics, verification code and dataset fingerprint."""
    return _models()["metadata"]


@app.post("/predict")
def predict(record: FarmRecord) -> dict:
    """Yield estimate, risk prediction, probability and cluster for one farm."""
    _models()
    try:
        return predict_record(record.model_dump())
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)

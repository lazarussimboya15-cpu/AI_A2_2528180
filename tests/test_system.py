"""Pytest suite: happy path AND failure paths."""
import copy
import importlib.util
import json

import pytest

from src.config import DEFAULT_STUDENT_ID, ROOT
from src.inference import InputValidationError, predict_record, validate_record
from src.pipeline import run_pipeline
from src.validation import ValidationError, check_no_leakage, validate_seed

SEED = 8180
VALID = json.loads((ROOT / "samples" / "valid.json").read_text())


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Use submitted models; train them if the folder was cleaned."""
    if not (ROOT / "models" / "metadata.json").exists():
        run_pipeline(DEFAULT_STUDENT_ID, SEED, verbose=False)


def test_valid_inference_returns_all_fields():
    out = predict_record(VALID)
    assert out["verification_code"] == "AI-A2-8180"
    assert out["yield_estimate_tons"] > 0
    assert out["high_risk_prediction"] in (0, 1)
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)


def test_missing_field_is_rejected():
    bad = copy.deepcopy(VALID)
    del bad["field_size_ha"]
    with pytest.raises(InputValidationError, match="field_size_ha"):
        predict_record(bad)


def test_wrong_type_is_rejected():
    bad = {**VALID, "rainfall_mm": "lots"}
    with pytest.raises(InputValidationError, match="finite number"):
        validate_record(bad)
    with pytest.raises(InputValidationError):
        validate_record({**VALID, "cooperative_support": True})


def test_range_error_is_rejected():
    with pytest.raises(InputValidationError, match="outside allowed range"):
        validate_record({**VALID, "avg_temp_c": 40})
    with pytest.raises(InputValidationError, match="season"):
        validate_record({**VALID, "season": "C"})


def test_wrong_seed_is_rejected():
    assert validate_seed(DEFAULT_STUDENT_ID, SEED) == "AI-A2-8180"
    with pytest.raises(ValidationError, match="does not match"):
        validate_seed(DEFAULT_STUDENT_ID, 1234)


def test_leakage_check_blocks_targets():
    with pytest.raises(ValidationError, match="leakage"):
        check_no_leakage(["rainfall_mm", "yield_tons"])


def test_reproducibility_same_fingerprint_and_metrics(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sa = run_pipeline(DEFAULT_STUDENT_ID, SEED, root=a, verbose=False)
    sb = run_pipeline(DEFAULT_STUDENT_ID, SEED, root=b, verbose=False)
    assert sa["fingerprint"] == sb["fingerprint"]
    ma = json.loads((a / "artifacts" / "regression_metrics.json").read_text())
    mb = json.loads((b / "artifacts" / "regression_metrics.json").read_text())
    for name in ma["models"]:
        assert ma["models"][name]["RMSE"] == pytest.approx(mb["models"][name]["RMSE"])
    assert sa["fingerprint"] == (ROOT / "artifacts" / "data_fingerprint.txt").read_text().split()[0]


def test_dataset_has_required_problems():
    spec = importlib.util.spec_from_file_location("g", ROOT / "data" / "generate_data.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    df = g.generate_dataframe(SEED)
    assert len(df) == 240 + SEED % 41 + 3 == 264
    assert df["soil_moisture_pct"].isna().sum() >= 13
    assert df.duplicated().sum() == 3


def test_api_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model-info").json()["verification_code"] == "AI-A2-8180"
    ok = client.post("/predict", json=VALID)
    assert ok.status_code == 200 and ok.json()["verification_code"] == "AI-A2-8180"
    bad = client.post("/predict", json={**VALID, "rainfall_mm": 5000})
    assert bad.status_code == 422
    missing = client.post("/predict", json={"season": "A"})
    assert missing.status_code == 422

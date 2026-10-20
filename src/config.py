"""Shared constants for the Musanze Harvest Intelligence Service."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUDENT_NAME = "Lazarus Simboya Ira Inyasio"
REG_NUMBER = "25/28180"
REG_DIGITS = "2528180"
DEFAULT_STUDENT_ID = "ug2528180@ines.ac.rw"

NUMERIC_FEATURES = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_REG = "yield_tons"
TARGET_CLF = "high_risk"
ALL_COLUMNS = FEATURES + [TARGET_REG, TARGET_CLF]

# Valid ranges: identical to the data-generation ranges.
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASONS = ("A", "B")
MIN_YIELD = 0.5
TEST_SIZE = 0.2
DISCLAIMER = (
    "Classroom decision-support prototype trained on synthetic data. It does not "
    "replace an agronomist and must not decide access to finance, insurance or "
    "public services. Human review is required."
)

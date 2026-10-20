"""Reproducible synthetic data generator (individual seed = last four digits).

Usage:  python data/generate_data.py --student-id ug2428180@ines.ac.rw --seed 8180
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def base_row_count(seed: int) -> int:
    """240 + (LLLL mod 41) rows before duplicates."""
    return 240 + (seed % 41)


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the raw dataset: base rows, missing values, three exact duplicates."""
    rng = np.random.default_rng(seed)
    n = base_row_count(seed)

    df = pd.DataFrame({
        "rainfall_mm": rng.uniform(600, 1400, n),
        "avg_temp_c": rng.uniform(12, 26, n),
        "soil_moisture_pct": rng.uniform(20, 80, n),
        "fertilizer_kg": rng.uniform(0, 250, n),
        "field_size_ha": rng.uniform(0.2, 4.0, n),
        "season": rng.choice(["A", "B"], n),
        "cooperative_support": rng.integers(0, 2, n),
    })
    season_effect = np.where(df["season"] == "A", 0.50, -0.20)
    noise = rng.normal(0.0, 0.80, n)
    y = (0.004 * df["rainfall_mm"] + 0.06 * df["soil_moisture_pct"]
         + 0.02 * df["fertilizer_kg"] + 1.30 * df["field_size_ha"]
         - 0.25 * (df["avg_temp_c"] - 18).abs() + 0.70 * df["cooperative_support"]
         + season_effect + noise)
    df["yield_tons"] = np.clip(y, 0.50, None)
    yield_per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (yield_per_ha < yield_per_ha.median()).astype(int)

    m = round(0.05 * n)
    df.loc[rng.choice(n, m, replace=False), "soil_moisture_pct"] = np.nan
    df.loc[rng.choice(n, m, replace=False), "fertilizer_kg"] = np.nan
    dup_pos = rng.choice(n, 3, replace=False)
    return pd.concat([df, df.iloc[dup_pos]], ignore_index=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(seed: int, csv_path: Path, fingerprint_path: Path) -> str:
    """Write the CSV (4 decimals) and its SHA-256 fingerprint; return the hash."""
    csv_path, fingerprint_path = Path(csv_path), Path(fingerprint_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataframe(seed).to_csv(csv_path, index=False, float_format="%.4f",
                                    lineterminator="\n")
    digest = sha256_file(csv_path)
    fingerprint_path.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    digits = "".join(re.findall(r"\d+", a.student_id.split("@")[0]))
    if not digits or int(digits[-4:]) != a.seed:
        raise SystemExit("Seed must equal the last four digits of the student ID.")
    out = root / "data" / f"AI_A2_{digits}.csv"
    print("SHA-256:", write_dataset(a.seed, out, root / "artifacts" / "data_fingerprint.txt"))


if __name__ == "__main__":
    main()

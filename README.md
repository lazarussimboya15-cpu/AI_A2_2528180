# Musanze Harvest Intelligence Service (AI Assignment 2)

Name: Lazarus Simboya Ira Inyasio |
Registration number: 25/28180 (digits 2528180) 
Institutional username: ug2528180@ines.ac.rw 
Verification code: AI-A2-8180 

**The dataset is synthetic.** It is generated from a formula plus random noise; no real farm data is used.
This is a classroom decision-support prototype. It does not replace an agronomist and must not decide
access to finance, insurance or public services. A human must review every output.

## Problem
A potato cooperative in Musanze needs to (1) estimate harvest tons, (2) flag high-production-risk
farms (yield per hectare below the median) and (3) group farms with similar conditions.

## Setup and run (tested: Python 3.12.3, Ubuntu 24 / Linux)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --student-id ug2528180@ines.ac.rw --seed 8180
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json     # prints errors, exit code 2
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000      # docs at http://127.0.0.1:8000/docs
```
Optional: add `--report` to `run_all.py` to rebuild `docs/AI_A2_2528180_REPORT.pdf`.

## Individual signature
Seed 8180 -> 261 base rows + 3 duplicates = 264 raw rows; 13 values set missing in each of
`soil_moisture_pct` and `fertilizer_kg`. SHA-256 of `data/AI_A2_2528180.csv`:
`6e4324acd04d7c209b33860dc6eded0f80048de0657a488bf124f65067644aae`
(also in `artifacts/data_fingerprint.txt`). `run_all.py` rejects any seed that is not 8180.

## Layout
`run_all.py` entry point | `predict.py` CLI | `app.py` FastAPI | `data/generate_data.py` generator |
`src/` config, validation, preprocessing, modeling, clustering, evaluation, inference, pipeline, report |
`models/` saved pipelines + `metadata.json` | `artifacts/` metrics JSON, `test_predictions.csv`, plots |
`tests/` pytest | `docs/` report | `evidence/` AI use, demo video, test log, declaration.

## Artifacts
`regression_metrics.json`, `classification_metrics.json`, `clustering_report.json`,
`data_validation_report.json`, `data_fingerprint.txt`, `test_predictions.csv`,
`plots/residuals.png`, `plots/confusion_matrix.png`, `plots/clusters.png`.

## API examples
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid.json
```
Invalid input (wrong type, out-of-range value, unknown or missing field) returns HTTP 422 with details.
Missing model files return HTTP 503.

## Extra packages
`reportlab` (build the PDF report), `httpx` (needed by FastAPI's TestClient), `uvicorn` (serve the API).

## Known limitations
Synthetic, independent features; small test set (53 rows); clusters are weak (silhouette about 0.19);
classification is easy because field size strongly determines yield per hectare in the formula.

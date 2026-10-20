"""Build docs/AI_A2_<REGDIGITS>_REPORT.pdf from the generated artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, Preformatted)

from .config import (DEFAULT_STUDENT_ID, DISCLAIMER, REG_DIGITS, REG_NUMBER, ROOT,
                     STUDENT_NAME)

SS = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontSize=10, leading=13.5)
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3)
CODE = ParagraphStyle("c", fontName="Courier", fontSize=7.8, leading=9.6)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8.5, leading=11)


def _tbl(rows, widths=None, font=8.6):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8dc")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    return t


def _p(text, style=BODY):
    return Paragraph(text, style)


def build_report(root: Path = ROOT) -> Path:
    root = Path(root)
    art = root / "artifacts"
    L = lambda n: json.loads((art / n).read_text(encoding="utf-8"))  # noqa: E731
    reg, clf, clu, val = (L("regression_metrics.json"), L("classification_metrics.json"),
                          L("clustering_report.json"), L("data_validation_report.json"))
    meta = json.loads((root / "models" / "metadata.json").read_text(encoding="utf-8"))
    code, seed, sha = val["verification_code"], val["seed"], val["sha256"]
    rb, cb = reg["selected_model"], clf["selected_model"]
    r, c = reg["models"][rb], clf["models"][cb]
    tn, fp, fn, tp = c["confusion_matrix"]["TN_FP_FN_TP"]
    q, cl = val["quality_raw"], val["cleaning"]
    out = root / "docs" / f"AI_A2_{REG_DIGITS}_REPORT.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                            title="AI Assignment 2 Technical Report", author=STUDENT_NAME)
    S = []

    # Page 1
    S += [_p("Musanze Harvest Intelligence Service - Technical Report", H1),
          _p("INES Ruhengeri | SWE 3513 Artificial Intelligence | Individual Assignment 2", SMALL),
          Spacer(1, 6),
          _tbl([["Item", "Value"], ["Student", STUDENT_NAME], ["Registration number", REG_NUMBER],
                ["Institutional username", DEFAULT_STUDENT_ID.split('@')[0]],
                ["Verification code", code], ["Seed (LLLL)", str(seed)],
                ["Dataset SHA-256", _p(sha[:32] + "<br/>" + sha[32:], SMALL)]], [4.5 * cm, 12.5 * cm]),
          _p("1. Identity and decision problem", H1),
          _p("<b>Stakeholder.</b> A field officer at a potato cooperative in Musanze who receives "
             "farm records in inconsistent spreadsheets and needs a local, offline-capable tool."),
          _p("<b>Prediction questions.</b> (1) Regression: how many tons of potatoes will a farm "
             "harvest? (2) Classification: is the farm at <i>high production risk</i> (yield per hectare "
             "below the median)? (3) Clustering: which farms share similar growing conditions?"),
          _p("<b>Ethical boundary.</b> " + DISCLAIMER + " Predictions are advice for a human "
             "officer to review, never an automatic decision about a farmer."),
          _p("<b>Outputs.</b> Command-line tool (<font face='Courier'>predict.py</font>) and a FastAPI "
             "service (<font face='Courier'>/health</font>, <font face='Courier'>/model-info</font>, "
             "<font face='Courier'>/predict</font>) for a future mobile app."), PageBreak()]

    # Page 2
    S += [_p("2. Data and reproducibility", H1),
          _p(f"The seed is the last four digits of the registration number ({seed}); "
             f"<font face='Courier'>numpy.random.default_rng({seed})</font> drives every random step. "
             f"Base rows = 240 + ({seed} mod 41) = <b>{val['base_rows_expected']}</b>. "
             f"The generator then set m = round(0.05 x {val['base_rows_expected']}) = "
             f"<b>{round(0.05 * val['base_rows_expected'])}</b> values missing in each of two columns and "
             "appended three exact duplicate rows."),
          _tbl([["Column", "Rule / role"],
                ["rainfall_mm", "Uniform 600-1400, feature"], ["avg_temp_c", "Uniform 12-26, feature"],
                ["soil_moisture_pct", "Uniform 20-80, feature (missing injected)"],
                ["fertilizer_kg", "Uniform 0-250, feature (missing injected)"],
                ["field_size_ha", "Uniform 0.2-4.0, feature"], ["season", "A or B, categorical feature"],
                ["cooperative_support", "0 or 1, binary feature"],
                ["yield_tons", "Formula + N(0, 0.80) noise, clipped at 0.5; regression target"],
                ["high_risk", "Yield per hectare below median; classification target"]],
               [4.5 * cm, 12.5 * cm]),
          Spacer(1, 6), _p("Data-quality evidence", H2),
          _tbl([["Check", "Result"],
                ["Raw rows", f"{q['rows_raw']} (= {val['base_rows_expected']} base + 3 duplicates)"],
                ["Missing values (raw file)", ", ".join(f"{k}: {v}" for k, v in q["missing_by_column"].items())],
                ["Exact duplicates detected", str(q["duplicate_rows"])],
                ["Duplicates removed", str(cl["duplicates_removed"])],
                ["Clean rows used for modelling", str(cl["rows_after"])],
                ["Range / schema check", "passed"], ["Leakage check", "passed"]], [6 * cm, 11 * cm]),
          _p("The raw file shows 13 missing values per column rather than 12 because one of the three "
             "duplicated rows copied a row that already had a missing value; removing duplicates restores "
             "12 per column.", SMALL),
          Spacer(1, 4), _p("SHA-256 fingerprint of the CSV bytes (artifacts/data_fingerprint.txt)", H2),
          Preformatted(sha, CODE),
          _p("Running <font face='Courier'>run_all.py</font> twice reproduces the same fingerprint "
             "because the CSV is written with four decimals and Unix line endings from the seeded generator; "
             "a test checks this.", BODY), PageBreak()]

    # Page 3
    S += [_p("3. Pipeline design", H1),
          _p("<b>Split.</b> One stratified 80/20 split with <font face='Courier'>random_state="
             f"{seed}</font>: {reg['n_train']} training and {reg['n_test']} test rows. The test set is used "
             "only for final reporting."),
          _p("<b>Preprocessing (one ColumnTransformer, fitted on training rows only).</b> Numeric features: "
             "median imputation then standard scaling. Season: one-hot encoding. cooperative_support: "
             "passed through. The same transformer lives inside every saved model Pipeline, so "
             "<font face='Courier'>predict.py</font> and the API repeat exactly the training transformations."),
          _p("<b>Leakage controls.</b> yield_tons, high_risk and yield per hectare are never features "
             "(<font face='Courier'>check_no_leakage</font> stops the run otherwise). "
             "Duplicates were removed before the split so a farm cannot appear in both train and test."),
          _p("<b>Candidates and selection.</b> Each task has a baseline and three candidates. Selection "
             "uses 5-fold cross-validation inside the training set (RMSE for regression, F1 for "
             "classification); test scores are computed afterwards and not used to choose."),
          _tbl([["Task", "Baseline", "Candidates", "Selected"],
                ["Regression", "mean", "linear regression, random forest, gradient boosting", rb],
                ["Classification", "majority class", "logistic regression, random forest, gradient boosting", cb],
                ["Clustering", "-", f"KMeans, K=2..6, silhouette", f"K={clu['selected_k']}"]],
               [2.8 * cm, 2.6 * cm, 8.4 * cm, 3.2 * cm]),
          Spacer(1, 6),
          _p("<b>Why a simple model can win.</b> The target formula is almost linear in the inputs, so "
             "linear and logistic models are strong candidates; tree ensembles need more data to match them "
             f"with only {reg['n_train']} training rows."),
          _p("<b>Determinism.</b> Every estimator, the split, the cross-validation folds and KMeans use "
             f"random_state={seed}. Versions: scikit-learn {meta['sklearn_version']}, "
             f"Python {meta['python_version']}."), PageBreak()]

    # Page 4
    rows = [["Model", "MAE", "RMSE", "R2", "CV RMSE (train)"]]
    for k, v in reg["models"].items():
        rows.append([k + (" *" if k == rb else ""), f"{v['MAE']:.4f}", f"{v['RMSE']:.4f}", f"{v['R2']:.4f}", f"{v['cv_RMSE_train']:.4f}"])
    crow = [["Model", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 (train)"]]
    for k, v in clf["models"].items():
        crow.append([k + (" *" if k == cb else ""), f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['F1']:.3f}",
                     "n/a" if v["ROC_AUC"] is None else f"{v['ROC_AUC']:.3f}", f"{v['cv_F1_train']:.3f}"])
    re_, ce = reg["important_error"], clf["important_error"]
    S += [_p("4. Regression and classification (held-out test set, * = selected)", H1),
          _p("Regression", H2), _tbl(rows, [5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3.4 * cm]),
          Spacer(1, 4),
          Image(str(art / "plots" / "residuals.png"), width=11.5 * cm, height=4.6 * cm),
          _p(f"<b>Residuals / selection.</b> {reg['selection_reason']}"),
          _p(f"<b>Important error (row {re_['row_id']}).</b> Actual {re_['actual_yield_tons']:.2f} t, predicted "
             f"{re_['predicted_yield_tons']:.2f} t (error {re_['error_tons']:+.2f} t, "
             f"{re_['error_in_noise_sd']:.1f} noise standard deviations). Because the data contains random "
             "noise with sd 0.80 t, the best possible RMSE is near 0.8 t; errors this size are mostly irreducible noise, not a model fault."),
          _p("Classification (positive class = high risk)", H2),
          _tbl(crow, [4.6 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 3.2 * cm]),
          Spacer(1, 4),
          Image(str(art / "plots" / "confusion_matrix.png"), width=4.6 * cm, height=4.1 * cm),
          _p(f"<b>Confusion matrix ({cb}).</b> TN={tn}, FP={fp}, FN={fn}, TP={tp}. "
             f"{clf['selection_reason']}"),
          _p(f"<b>Important error (row {ce.get('row_id', '-')}).</b> "
             + (f"Actual class {ce['actual_high_risk']}, predicted {ce['predicted_high_risk']} "
                f"(probability {ce['probability_high_risk']:.2f}). Its yield per hectare was "
                f"{ce['actual_yield_per_ha']:.2f} versus a threshold of {ce['risk_threshold_yield_per_ha']:.2f}, "
                "so it lies close to the median cut-off where noise decides the label." if "row_id" in ce else ce["note"])),
          PageBreak()]

    # Page 5
    sil = clu["silhouette_by_k"]
    prow = [["Cluster", "Rows", "Rain", "Temp", "Soil m.", "Fert.", "Size ha", "Profile"]]
    for k, v in clu["profiles"].items():
        m = v["feature_means"]
        prow.append([k, v["size_train_rows"], f"{m['rainfall_mm']:.0f}", f"{m['avg_temp_c']:.1f}",
                     f"{m['soil_moisture_pct']:.0f}", f"{m['fertilizer_kg']:.0f}", f"{m['field_size_ha']:.1f}",
                     _p(v["summary"].split(": ", 1)[1], SMALL)])
    best_p = clu["profiles"][min(clu["profiles"], key=lambda k: -clu["profiles"][k]["size_train_rows"])]
    S += [_p("5. Clustering", H1),
          _p("KMeans was fitted on the five scaled numeric conditions of the training rows (targets unused). "
             "Season and cooperative support were left out of the distance because 0/1 columns made the first "
             "version simply split farms by season; they are used to describe clusters instead."),
          _tbl([["K"] + list(sil), ["Silhouette"] + [f"{v:.4f}" for v in sil.values()]], None),
          _p(f"<b>Selected K = {clu['selected_k']}</b> (highest silhouette, {clu['selected_silhouette']:.3f}). "
             f"{clu['interpretation']}"), Spacer(1, 4),
          Image(str(art / "plots" / "clusters.png"), width=10 * cm, height=7.5 * cm), Spacer(1, 4),
          _tbl(prow, [1.4 * cm, 1.2 * cm, 1.3 * cm, 1.3 * cm, 1.5 * cm, 1.3 * cm, 1.5 * cm, 7.5 * cm], font=8),
          Spacer(1, 4),
          _p("<b>Operational meaning.</b> A field officer can use the cluster label to compare a farm with "
             "others that have similar rainfall, soil moisture, fertilizer and field size, for example to plan "
             "advice visits. Because silhouette scores are low, the clusters are convenience groupings, "
             "not distinct farm types, and should not drive decisions on their own."), PageBreak()]

    # Page 6
    S += [_p("6. API and testing", H1),
          _tbl([["Endpoint", "Purpose"],
                ["GET /health", "Service status, models loaded, verification code"],
                ["GET /model-info", "Selected models, test metrics, fingerprint, disclaimer"],
                ["POST /predict", "Typed farm record -> yield, risk, probability, cluster, code"]],
               [4 * cm, 13 * cm]),
          _p("Valid request", H2),
          Preformatted(json.dumps(json.loads((root / "samples" / "valid.json").read_text()), indent=1), CODE)]
    from .inference import predict_record
    pr = predict_record(json.loads((root / "samples" / "valid.json").read_text()), root / "models")
    short = {k: pr[k] for k in ("verification_code", "yield_estimate_tons", "high_risk_prediction",
                                 "high_risk_probability", "cluster")}
    S += [_p("Response (CLI and API return the same fields)", H2),
          Preformatted(json.dumps(short, indent=1), CODE),
          _p("Invalid request and failure behaviour", H2),
          _p("The request is validated by typed Pydantic fields (<font face='Courier'>extra='forbid'</font>, "
             "min/max ranges equal to the training ranges). A bad value, unknown field or missing field "
             "returns HTTP 422 with a list of problems; missing model files return 503. "
             "<font face='Courier'>predict.py</font> prints the same problems and exits with code 2. "
             "Example: rainfall_mm=2500 gives 'outside allowed range [600.0, 1400.0]'."),
          _p("Automated tests (tests/test_system.py)", H2),
          _tbl([["Test", "Covers"],
                ["valid_inference_returns_all_fields", "valid inference"],
                ["missing_field_is_rejected", "missing field"], ["wrong_type_is_rejected", "wrong type"],
                ["range_error_is_rejected", "range error"],
                ["wrong_seed_is_rejected / leakage", "seed rule, target leakage"],
                ["reproducibility_same_fingerprint_and_metrics", "two runs, same SHA-256 and metrics"],
                ["dataset_has_required_problems", "rows, missing values, duplicates"],
                ["api_endpoints", "/health, /model-info, /predict 200 and 422"]], [7.5 * cm, 9.5 * cm]),
          _p("Run <font face='Courier'>pytest -q</font> and attach the output as evidence/..._TEST_LOG.pdf.", SMALL),
          PageBreak()]

    # Page 7
    S += [_p("7. Model card and reflection", H1),
          _p(f"<b>Intended use.</b> Classroom prototype that helps a field officer estimate harvest, flag "
             "possible high-risk farms and group similar farms."),
          _p("<b>Synthetic data.</b> All records are generated from a formula with random noise; no real farm "
             "data was used. Metrics describe how well the models recover this formula, not real Musanze farms."),
          _p("<b>Limitations.</b> Only seven inputs; uniform independent features; no weather extremes, pests, "
             "prices or soil type; ranges limited to 600-1400 mm rainfall and 0.2-4 ha fields; the test set has "
             f"only {reg['n_test']} rows so scores vary; clusters are weak (silhouette {clu['selected_silhouette']:.2f})."),
          _p("<b>Risks.</b> Over-trust in a score; a wrong 'low risk' result delaying support; unfair treatment "
             "of small farms if used outside its purpose; applying the model to farms unlike the training data."),
          _p("<b>Prohibited use.</b> Deciding access to loans, insurance, subsidies or public services; "
             "replacing an agronomist; use on real farmers' data without validation and consent."),
          _p("<b>Human review.</b> Every output must be reviewed by a qualified person who can override it."),
          _p("<b>With real data I would</b> collect cooperative records with consent, add soil type, variety and "
             "pest information, validate on a later season (time-based split), check performance by "
             "season and farm size, calibrate probabilities, and choose the risk threshold with the cooperative."),
          _p(DISCLAIMER, SMALL)]

    def footer(canvas, d):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.9 * cm, 1.0 * cm, f"{STUDENT_NAME} | {REG_NUMBER} | {code}")
        canvas.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"Page {d.page}")

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return out

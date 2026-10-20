"""End-to-end pipeline: data -> validation -> cleaning -> 3 ML tasks -> artifacts."""
from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import clustering, evaluation, modeling, preprocessing, validation
from .config import (DEFAULT_STUDENT_ID, DISCLAIMER, FEATURES, RANGES, ROOT,
                     TARGET_CLF, TARGET_REG)
from .utils import write_json


def _load_generator(root: Path):
    spec = importlib.util.spec_from_file_location("generate_data",
                                                  Path(__file__).resolve().parents[1] / "data" / "generate_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pipeline(student_id: str, seed: int, root: Path = ROOT, verbose: bool = True) -> dict:
    """Run everything and write all artifacts under ``root``."""
    root = Path(root)
    log = print if verbose else (lambda *a, **k: None)
    code = validation.validate_seed(student_id, seed)
    digits = validation.student_digits(student_id)
    log(f"Verification code: {code}")

    art, mdl = root / "artifacts", root / "models"
    for d in (art, art / "plots", mdl, root / "data"):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data generation + fingerprint
    gen = _load_generator(root)
    csv_path = root / "data" / f"AI_A2_{digits}.csv"
    fingerprint = gen.write_dataset(seed, csv_path, art / "data_fingerprint.txt")
    log(f"Dataset: {csv_path.name}  SHA-256: {fingerprint}")

    # 2. Validation + cleaning
    raw = pd.read_csv(csv_path)
    validation.validate_schema(raw)
    validation.validate_ranges(raw)
    quality = validation.data_quality_report(raw)
    df, cleaning = preprocessing.clean_dataset(raw)
    validation.check_no_leakage(FEATURES)
    write_json(art / "data_validation_report.json", {
        "verification_code": code, "student_id": student_id, "seed": seed,
        "base_rows_expected": gen.base_row_count(seed), "sha256": fingerprint,
        "quality_raw": quality, "cleaning": cleaning,
        "imputation": "median, fitted inside each Pipeline on training rows only",
        "leakage_check": "passed: yield_tons, high_risk and yield per hectare are not features"})
    log(f"Rows raw={quality['rows_raw']} missing={quality['missing_by_column']} "
        f"duplicates_removed={cleaning['duplicates_removed']} clean={cleaning['rows_after']}")

    # 3. Split (train-only fitting from here on)
    train_idx, test_idx = preprocessing.split_indices(df, seed)
    train, test = df.loc[train_idx], df.loc[test_idx]
    Xtr, Xte = train[FEATURES], test[FEATURES]

    # 4. Regression
    reg_res, reg_best, reg_reason, reg_model = modeling.run_regression(
        Xtr, train[TARGET_REG], Xte, test[TARGET_REG], seed)
    reg_pred = reg_model.predict(Xte)
    reg_case = evaluation.regression_error_case(test, reg_pred)
    write_json(art / "regression_metrics.json", {
        "verification_code": code, "seed": seed,
        "n_train": len(train), "n_test": len(test), "features": FEATURES,
        "models": reg_res, "selected_model": reg_best, "selection_reason": reg_reason,
        "important_error": reg_case})

    # 5. Classification
    clf_res, clf_best, clf_reason, clf_model = modeling.run_classification(
        Xtr, train[TARGET_CLF], Xte, test[TARGET_CLF], seed)
    clf_pred = clf_model.predict(Xte)
    clf_prob = clf_model.predict_proba(Xte)[:, 1]
    threshold = float((df[TARGET_REG] / df["field_size_ha"]).median())
    clf_case = evaluation.classification_error_case(test, clf_pred, clf_prob, threshold)
    write_json(art / "classification_metrics.json", {
        "verification_code": code, "seed": seed, "positive_class": "high_risk = 1",
        "n_train": len(train), "n_test": len(test), "features": FEATURES,
        "target_leakage_check": "passed",
        "models": clf_res, "selected_model": clf_best, "selection_reason": clf_reason,
        "important_error": clf_case})

    # 6. Clustering (targets unused)
    scores, best_k, cl_model = clustering.fit_clustering(Xtr, seed)
    profiles = clustering.describe_clusters(cl_model, Xtr)
    test_clusters = cl_model.predict(Xte)
    sil = scores[best_k]
    strength = ("weak" if sil < 0.25 else "moderate" if sil < 0.5 else "strong")
    write_json(art / "clustering_report.json", {
        "verification_code": code, "seed": seed,
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": best_k, "selected_silhouette": sil,
        "cluster_structure_strength": strength,
        "profiles": {str(k): v for k, v in profiles.items()},
        "interpretation": (
            f"K={best_k} has the highest silhouette ({sil:.3f}), which indicates {strength} "
            "cluster structure. Input features were generated independently, so clusters "
            "are rough groupings of similar farm conditions, not natural farm types. "
            "Targets were not used to fit clusters.")})

    # 7. Test predictions CSV
    out = pd.DataFrame({
        "row_id": test.index,
        "actual_yield_tons": test[TARGET_REG].to_numpy(),
        "predicted_yield_tons": reg_pred,
        "yield_error_tons": reg_pred - test[TARGET_REG].to_numpy(),
        "abs_yield_error_tons": np.abs(reg_pred - test[TARGET_REG].to_numpy()),
        "actual_high_risk": test[TARGET_CLF].to_numpy(),
        "predicted_high_risk": clf_pred,
        "probability_high_risk": clf_prob,
        "risk_correct": (clf_pred == test[TARGET_CLF].to_numpy()).astype(int),
        "cluster": test_clusters})
    out.to_csv(art / "test_predictions.csv", index=False, float_format="%.4f",
               lineterminator="\n")

    # 8. Plots
    evaluation.plot_residuals(test[TARGET_REG], reg_pred, art / "plots" / "residuals.png", code)
    evaluation.plot_confusion(clf_res[clf_best]["confusion_matrix"]["matrix"],
                              art / "plots" / "confusion_matrix.png", code, clf_best)
    all_clusters = cl_model.predict(df[FEATURES])
    evaluation.plot_clusters(cl_model, df, all_clusters, art / "plots" / "clusters.png", code)

    # 9. Save models + metadata
    joblib.dump(reg_model, mdl / "regressor.joblib")
    joblib.dump(clf_model, mdl / "classifier.joblib")
    joblib.dump(cl_model, mdl / "clusterer.joblib")
    meta = {
        "verification_code": code, "student_id": student_id, "seed": seed,
        "dataset_sha256": fingerprint, "features": FEATURES,
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "regression_model": reg_best, "classification_model": clf_best,
        "n_clusters": best_k,
        "cluster_profiles": {str(k): v["summary"] for k, v in profiles.items()},
        "regression_test": reg_res[reg_best], "classification_test": {
            k: v for k, v in clf_res[clf_best].items() if k != "confusion_matrix"},
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "disclaimer": DISCLAIMER}
    write_json(mdl / "metadata.json", meta)

    summary = {"verification_code": code, "fingerprint": fingerprint, "regression": reg_best,
               "classification": clf_best, "k": best_k,
               "artifacts": sorted(str(p.relative_to(root)) for p in
                                   list(art.rglob("*.*")) + list(mdl.glob("*")))}
    if verbose:
        r, c = reg_res[reg_best], clf_res[clf_best]
        log(f"Regression  [{reg_best}] MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.4f}")
        log(f"Classifier  [{clf_best}] P={c['precision']:.4f} R={c['recall']:.4f} "
            f"F1={c['F1']:.4f} AUC={c['ROC_AUC']:.4f}")
        log(f"Clustering  K={best_k} silhouette={sil:.4f}")
        log("Artifacts:\n  " + "\n  ".join(summary["artifacts"]))
    return summary

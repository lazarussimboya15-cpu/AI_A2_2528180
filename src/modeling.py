"""Regression and classification: baselines, candidates, CV-based selection."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

BASELINE_REG = "mean_baseline"
BASELINE_CLF = "majority_baseline"


def _wrap(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def regression_candidates(seed: int) -> dict:
    return {
        BASELINE_REG: DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def classification_candidates(seed: int) -> dict:
    return {
        BASELINE_CLF: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:  # only one class present
        auc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "ROC_AUC": auc,
            "confusion_matrix": {"labels": ["low_risk(0)", "high_risk(1)"],
                                 "matrix": cm.tolist(),
                                 "TN_FP_FN_TP": [int(v) for v in cm.ravel()]}}


def run_regression(X_train, y_train, X_test, y_test, seed: int):
    """Fit all candidates on train; select by 5-fold CV RMSE on train only."""
    cv = KFold(5, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, est in regression_candidates(seed).items():
        pipe = _wrap(est)
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_train, y_train)
        results[name] = {**regression_metrics(y_test, pipe.predict(X_test)),
                         "cv_RMSE_train": float(cv_rmse)}
        fitted[name] = pipe
    cands = {k: v for k, v in results.items() if k != BASELINE_REG}
    best = min(cands, key=lambda k: cands[k]["cv_RMSE_train"])
    reason = (f"{best} had the lowest 5-fold cross-validated RMSE on the training set "
              f"({cands[best]['cv_RMSE_train']:.4f} tons) and beats the mean baseline "
              f"(test RMSE {results[BASELINE_REG]['RMSE']:.4f} vs {results[best]['RMSE']:.4f}). "
              "Selection used training-set CV only; test scores were reported afterwards.")
    return results, best, reason, fitted[best]


def run_classification(X_train, y_train, X_test, y_test, seed: int):
    """Fit all candidates on train; select by 5-fold CV F1 on train only."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, est in classification_candidates(seed).items():
        pipe = _wrap(est)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {**classification_metrics(y_test, pred, prob),
                         "cv_F1_train": float(cv_f1)}
        fitted[name] = pipe
    cands = {k: v for k, v in results.items() if k != BASELINE_CLF}
    best = max(cands, key=lambda k: cands[k]["cv_F1_train"])
    reason = (f"{best} had the highest 5-fold cross-validated F1 on the training set "
              f"({cands[best]['cv_F1_train']:.4f}). The majority baseline has test F1 "
              f"{results[BASELINE_CLF]['F1']:.4f} and recall {results[BASELINE_CLF]['recall']:.4f}. "
              "Selection used training CV only; test scores were reported afterwards.")
    return results, best, reason, fitted[best]

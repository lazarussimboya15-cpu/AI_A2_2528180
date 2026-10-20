"""KMeans clustering on scaled features only (no targets), K = 2..6."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from .config import FEATURES, NUMERIC_FEATURES

LABELS = {"rainfall_mm": "rainfall", "avg_temp_c": "temperature",
          "soil_moisture_pct": "soil moisture", "fertilizer_kg": "fertilizer use",
          "field_size_ha": "field size"}


def make_cluster_pipeline(k: int, seed: int) -> Pipeline:
    """Cluster on the five scaled numeric conditions.

    Season and cooperative support are left out of the distance calculation: as 0/1
    columns they would dominate and simply split farms by season. They are used to
    describe the clusters afterwards instead.
    """
    prep = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())]), NUMERIC_FEATURES)])
    return Pipeline([("prep", prep),
                     ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))])


def fit_clustering(X_train: pd.DataFrame, seed: int):
    """Score K=2..6 by silhouette on the training rows; return scores, best K, model."""
    scores = {}
    for k in range(2, 7):
        pipe = make_cluster_pipeline(k, seed).fit(X_train[FEATURES])
        Xt = pipe[:-1].transform(X_train[FEATURES])
        scores[k] = float(silhouette_score(Xt, pipe["kmeans"].labels_))
    best_k = max(scores, key=scores.get)
    model = make_cluster_pipeline(best_k, seed).fit(X_train[FEATURES])
    return scores, best_k, model


def describe_clusters(model: Pipeline, X_train: pd.DataFrame) -> dict:
    """Plain-language profile per cluster from raw feature means (targets unused)."""
    labels = model["kmeans"].labels_
    overall_mean = X_train[NUMERIC_FEATURES].mean()
    overall_std = X_train[NUMERIC_FEATURES].std()
    profiles = {}
    for c in sorted(set(labels)):
        rows = X_train[labels == c]
        means = rows[NUMERIC_FEATURES].mean()
        z = (means - overall_mean) / overall_std
        parts = [f"{'higher' if z[f] > 0 else 'lower'} {LABELS[f]}"
                 for f in z.abs().sort_values(ascending=False).index if abs(z[f]) >= 0.35][:3]
        share_a = float((rows["season"] == "A").mean())
        support = float(rows["cooperative_support"].mean())
        parts.append("mostly season A" if share_a >= 0.65 else
                     "mostly season B" if share_a <= 0.35 else "mixed seasons")
        parts.append("mostly supported by the cooperative" if support >= 0.65 else
                     "mostly without cooperative support" if support <= 0.35
                     else "mixed cooperative support")
        profiles[int(c)] = {
            "size_train_rows": int(len(rows)),
            "feature_means": {f: float(means[f]) for f in NUMERIC_FEATURES},
            "season_A_share": share_a,
            "cooperative_support_share": support,
            "summary": f"Cluster {c}: " + ", ".join(parts) + ".",
        }
    return profiles

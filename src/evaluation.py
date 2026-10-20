"""Plots and error analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .config import FEATURES  # noqa: E402


def plot_residuals(y_true, y_pred, path: Path, code: str) -> None:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_pred, resid, alpha=0.7)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set_xlabel("Predicted yield (tons)")
    ax[0].set_ylabel("Residual = actual - predicted (tons)")
    ax[0].set_title("Residuals vs prediction")
    ax[1].hist(resid, bins=12, edgecolor="black")
    ax[1].set_xlabel("Residual (tons)")
    ax[1].set_title("Residual distribution")
    fig.suptitle(f"Regression residuals - {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(matrix, path: Path, code: str, model_name: str) -> None:
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["low risk", "high risk"])
    ax.set_yticks([0, 1], ["low risk", "high risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name}\nTest confusion matrix - {code}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(cluster_model, X_all, clusters, path: Path, code: str) -> None:
    Xt = cluster_model[:-1].transform(X_all[FEATURES])
    pca = PCA(n_components=2, random_state=0).fit(Xt)
    pts = pca.transform(Xt)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=clusters, cmap="tab10", alpha=0.8)
    ax.legend(*sc.legend_elements(), title="Cluster")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.set_title(f"Farm clusters (PCA view) - {code}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def regression_error_case(test_df, y_pred) -> dict:
    """Largest absolute regression error on the test set, with context."""
    err = y_pred - test_df["yield_tons"].to_numpy()
    i = int(np.argmax(np.abs(err)))
    row = test_df.iloc[i]
    return {"row_id": int(test_df.index[i]),
            "actual_yield_tons": float(row["yield_tons"]),
            "predicted_yield_tons": float(y_pred[i]),
            "error_tons": float(err[i]),
            "error_in_noise_sd": float(abs(err[i]) / 0.80),
            "features": {k: (None if str(row[k]) == "nan" else
                             (row[k].item() if hasattr(row[k], "item") else row[k]))
                         for k in FEATURES},
            "note": "Noise has sd 0.80 t, so errors of 2+ sd are expected in a few rows "
                    "and cannot be removed by any model."}


def classification_error_case(test_df, pred, prob, threshold: float) -> dict:
    """Most confident wrong classification on the test set."""
    y = test_df["high_risk"].to_numpy()
    wrong = np.where(pred != y)[0]
    if len(wrong) == 0:
        return {"note": "No misclassified test rows."}
    conf = np.where(y[wrong] == 1, 1 - prob[wrong], prob[wrong])
    i = int(wrong[np.argmax(conf)])
    row = test_df.iloc[i]
    ypha = float(row["yield_tons"] / row["field_size_ha"])
    return {"row_id": int(test_df.index[i]),
            "actual_high_risk": int(y[i]), "predicted_high_risk": int(pred[i]),
            "probability_high_risk": float(prob[i]),
            "actual_yield_per_ha": ypha,
            "risk_threshold_yield_per_ha": float(threshold),
            "distance_to_threshold": float(ypha - threshold),
            "note": "Rows close to the median yield-per-hectare threshold are hard to "
                    "classify because random noise can push them across it."}

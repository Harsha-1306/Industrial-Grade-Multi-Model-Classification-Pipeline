"""
src/explainability/explainer.py
─────────────────────────────────────────────────────────────────
Model explainability without external XAI dependencies:
  • Permutation feature importance (model-agnostic)
  • Partial Dependence Plots (PDP) data
  • Decision boundary grids for 2-D projections
  • Individual prediction confidence breakdown
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

log = get_logger("Explainer")


def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                           feature_names: List[str],
                           n_repeats: int = 30,
                           seed: int = 42) -> Dict[str, Any]:
    """
    Model-agnostic permutation importance:
    Δ accuracy when each feature column is shuffled.
    Mirrors sklearn / SHAP importance semantics.
    """
    rng      = np.random.default_rng(seed)
    baseline = accuracy_score(y, model.predict(X))
    importances = np.zeros((n_repeats, X.shape[1]))

    for rep in range(n_repeats):
        for col in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            importances[rep, col] = baseline - accuracy_score(y, model.predict(X_perm))

    means = importances.mean(axis=0)
    stds  = importances.std(axis=0)
    order = np.argsort(means)[::-1]
    log.info(f"[Explainer] Top features: "
             + ", ".join(f"{feature_names[i]}={means[i]:.4f}" for i in order[:5]))
    return dict(
        feature_names = [feature_names[i] for i in order],
        importances   = means[order],
        stds          = stds[order],
        baseline_acc  = baseline,
        raw           = importances,
    )


def partial_dependence(model, X: np.ndarray, feature_idx: int,
                       n_grid: int = 50) -> Dict[str, np.ndarray]:
    """1-D Partial Dependence Plot data for a single feature."""
    grid   = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_grid)
    avg_preds = []
    for val in grid:
        X_tmp = X.copy()
        X_tmp[:, feature_idx] = val
        probs = model.predict_proba(X_tmp) if hasattr(model, "predict_proba") \
                else None
        if probs is not None:
            avg_preds.append(probs.mean(axis=0))
    return dict(grid=grid, avg_proba=np.array(avg_preds))


def prediction_confidence_breakdown(model, X: np.ndarray,
                                    class_names: List[str]) -> Dict:
    """Per-sample confidence (entropy) and top-class probability."""
    proba   = model.predict_proba(X)
    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    top_p   = proba.max(axis=1)
    pred    = proba.argmax(axis=1)
    return dict(
        proba=proba, entropy=entropy, top_p=top_p,
        pred=pred, class_names=class_names
    )

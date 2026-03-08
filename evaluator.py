"""
src/evaluation/evaluator.py
─────────────────────────────────────────────────────────────────
Comprehensive model evaluation:
  • Accuracy / F1 / Precision / Recall / MCC / Cohen's Kappa
  • Per-class metrics
  • Bootstrap confidence intervals
  • Calibration analysis
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

log = get_logger("Evaluator")


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=500, ci=0.95, seed=42):
    rng    = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    alpha = (1 - ci) / 2
    return (np.percentile(scores, alpha*100),
            np.percentile(scores, (1-alpha)*100))


def evaluate_model(name: str, model, X: np.ndarray,
                   y_true: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="weighted")
    prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = dict(
        accuracy=acc, f1_weighted=f1,
        precision=prec, recall=rec,
        mcc=mcc, kappa=kappa
    )

    if y_proba is not None:
        metrics["log_loss"] = log_loss(y_true, y_proba)
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
        except Exception:
            pass

    # Bootstrap CI on accuracy
    lo, hi = bootstrap_ci(y_true, y_pred,
                          lambda yt, yp: accuracy_score(yt, yp))
    metrics["acc_ci_lo"] = lo
    metrics["acc_ci_hi"] = hi

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    log.info(f"[Eval] {name:<25}  "
             f"acc={acc:.4f} [{lo:.4f},{hi:.4f}]  "
             f"f1={f1:.4f}  mcc={mcc:.4f}  kappa={kappa:.4f}")

    return dict(name=name, metrics=metrics, cm=cm,
                report=report, y_pred=y_pred, y_proba=y_proba)

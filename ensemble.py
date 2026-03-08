"""
src/models/ensemble.py
─────────────────────────────────────────────────────────────────
Industrial-grade ensemble layer:
  • Stacking with cross-val blending (no leakage)
  • Soft-voting ensemble
  • sklearn base estimators with calibrated probabilities
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import CFG
from src.utils.logger import get_logger

log = get_logger("EnsembleModels")


# ─────────────────────────────────────────────────────────────────────────────
# Base estimator zoo
# ─────────────────────────────────────────────────────────────────────────────
def get_base_estimators(seed: int = 42) -> Dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None,
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=seed
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "SVM_RBF": CalibratedClassifierCV(
            SVC(C=10, gamma="scale", kernel="rbf",
                class_weight="balanced", random_state=seed),
            cv=3, method="isotonic"
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=seed
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stacking ensemble
# ─────────────────────────────────────────────────────────────────────────────
class StackingEnsemble:
    """
    2-level stacking:
      L0 : diverse base models (OOF predictions used as L1 features)
      L1 : meta-learner trained on OOF probability stacks
    No leakage — L1 only sees out-of-fold predictions.
    """

    def __init__(self, base_models: Dict, meta_learner=None,
                 n_folds: int = 5, seed: int = 42):
        self.base_models  = base_models
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed
        )
        self.n_folds = n_folds
        self.seed    = seed
        self._fitted_bases: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, n_cls    = len(y), len(np.unique(y))
        kf          = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                      random_state=self.seed)
        oof_preds   = np.zeros((n, len(self.base_models) * n_cls))
        names       = list(self.base_models.keys())

        log.info(f"[Stacking] Generating OOF predictions for {len(names)} base models …")
        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            for m_i, (name, model) in enumerate(self.base_models.items()):
                import copy
                m = copy.deepcopy(model)
                m.fit(X[tr_idx], y[tr_idx])
                proba = m.predict_proba(X[val_idx])
                oof_preds[val_idx, m_i*n_cls:(m_i+1)*n_cls] = proba
            if (fold_i + 1) % 2 == 0:
                log.info(f"[Stacking] Fold {fold_i+1}/{self.n_folds} done")

        # Fit base models on full training data
        for name, model in self.base_models.items():
            model.fit(X, y)
            self._fitted_bases[name] = model
            log.info(f"[Stacking] {name} fitted on full train set")

        # Fit meta-learner
        self.meta_learner.fit(oof_preds, y)
        self.n_cls_ = n_cls
        log.info("[Stacking] Meta-learner fitted ✓")
        return self

    def _base_probas(self, X: np.ndarray) -> np.ndarray:
        n_cls   = self.n_cls_
        all_p   = []
        for name, model in self._fitted_bases.items():
            all_p.append(model.predict_proba(X))
        return np.hstack(all_p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict_proba(self._base_probas(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Soft-voting ensemble (weighted average of probabilities)
# ─────────────────────────────────────────────────────────────────────────────
class SoftVotingEnsemble:
    def __init__(self, models: Dict, weights: Optional[List[float]] = None):
        self.models  = models
        self.weights = weights or [1.0] * len(models)

    def fit(self, X, y):
        for name, m in self.models.items():
            m.fit(X, y)
            log.info(f"[SoftVoting] {name} fitted ✓")
        return self

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models.values()])
        w     = np.array(self.weights)[:, None, None]
        return (probs * w).sum(axis=0) / w.sum()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

"""
src/training/hpo.py
─────────────────────────────────────────────────────────────────
Hyperparameter optimisation pipeline:
  • RandomizedSearchCV with StratifiedKFold
  • Parallel execution (n_jobs=-1)
  • Best-model export
"""

from __future__ import annotations
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform, loguniform
from typing import Dict, Any, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import CFG
from src.utils.logger import get_logger

log = get_logger("HPO")


SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "RandomForest": {
        "estimator": RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=42),
        "params": {
            "n_estimators":    randint(50, 300),
            "max_depth":       [None, 5, 10],
            "min_samples_split": randint(2, 8),
            "min_samples_leaf":  randint(1, 4),
            "max_features":    ["sqrt", "log2"],
        }
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators":  randint(50, 200),
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth":     randint(2, 6),
            "subsample":     uniform(0.6, 0.4),
        }
    },
    "SVM": {
        "estimator": CalibratedClassifierCV(
            SVC(class_weight="balanced", random_state=42), cv=3, method="isotonic"
        ),
        "params": {
            "estimator__C":      loguniform(0.1, 50),
            "estimator__gamma":  loguniform(1e-4, 1.0),
            "estimator__kernel": ["rbf"],
        }
    },
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "params": {
            "C":       loguniform(0.01, 100),
            "solver":  ["lbfgs"],
            "penalty": ["l2"],
        }
    },
}


def run_hpo(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Run RandomizedSearchCV for all models, return dict of best estimators
    and their CV scores.
    """
    cfg = CFG.hpo
    cv  = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=CFG.seed)
    results: Dict[str, Any] = {}

    for name, space in SEARCH_SPACES.items():
        log.info(f"[HPO] Searching {name} …")
        search = RandomizedSearchCV(
            estimator  = space["estimator"],
            param_distributions = space["params"],
            n_iter     = cfg.n_iter,
            cv         = cv,
            scoring    = cfg.scoring,
            n_jobs     = cfg.n_jobs,
            random_state = CFG.seed,
            refit      = True,
            verbose    = 0,
        )
        search.fit(X_train, y_train)
        results[name] = {
            "best_estimator": search.best_estimator_,
            "best_params":    search.best_params_,
            "best_score":     search.best_score_,
            "cv_results":     search.cv_results_,
        }
        log.info(f"[HPO] {name}  best CV {cfg.scoring} = {search.best_score_:.4f}")
        log.info(f"[HPO] {name}  params = {search.best_params_}")

    return results

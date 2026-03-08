"""
src/data/pipeline.py
─────────────────────────────────────────────────────────────────
Production-grade data pipeline:
  • Loading & validation
  • Feature engineering
  • SMOTE-style minority class oversampling
  • Stratified multi-split (train / val / test)
  • sklearn-compatible custom transformers
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PolynomialFeatures, label_binarize
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from typing import Tuple, Dict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import CFG
from src.utils.logger import get_logger

log = get_logger("DataPipeline", CFG.LOG_DIR if hasattr(CFG, "LOG_DIR") else None)


# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer.
    Adds domain-inspired ratio and interaction features.
    Mirrors feature-engineering steps in industrial pipelines.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        sl, sw, pl, pw = X[:,0], X[:,1], X[:,2], X[:,3]
        eps = 1e-8
        # Ratio features (commonly used in bio-measurement tasks)
        petal_ratio   = pl / (pw + eps)
        sepal_ratio   = sl / (sw + eps)
        petal_area    = pl * pw
        sepal_area    = sl * sw
        # Interaction
        petal_sepal_diff = (pl + pw) - (sl + sw)
        return np.column_stack([
            X, petal_ratio, sepal_ratio,
            petal_area, sepal_area, petal_sepal_diff
        ])

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features else [f"f{i}" for i in range(4)]
        return base + [
            "petal_ratio", "sepal_ratio",
            "petal_area",  "sepal_area", "petal_sepal_diff"
        ]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise extreme values — IQR-based, fitted on train only."""
    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.array(X)
        q1  = np.percentile(X, 25, axis=0)
        q3  = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


# ─────────────────────────────────────────────────────────────────────────────
def smote_oversample(X: np.ndarray, y: np.ndarray,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight SMOTE-style oversampling (no imbalanced-learn dependency).
    Interpolates synthetic minority samples between existing neighbours.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    rng = np.random.default_rng(random_state)
    X_out, y_out = [X], [y]

    for cls, cnt in zip(classes, counts):
        if cnt == max_count:
            continue
        n_needed  = max_count - cnt
        X_cls     = X[y == cls]
        idx_a     = rng.integers(0, len(X_cls), size=n_needed)
        idx_b     = rng.integers(0, len(X_cls), size=n_needed)
        alphas    = rng.uniform(0, 1, size=(n_needed, 1))
        synthetic = X_cls[idx_a] * alphas + X_cls[idx_b] * (1 - alphas)
        X_out.append(synthetic)
        y_out.append(np.full(n_needed, cls))

    X_res = np.vstack(X_out)
    y_res = np.concatenate(y_out)
    perm  = rng.permutation(len(y_res))
    return X_res[perm], y_res[perm]


# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessing_pipeline() -> Pipeline:
    """Returns the full sklearn preprocessing pipeline."""
    return Pipeline([
        ("imputer",   SimpleImputer(strategy="median")),   # handles missing
        ("clipper",   OutlierClipper(factor=3.0)),
        ("engineer",  FeatureEngineer()),
        ("scaler",    RobustScaler()),                     # robust to outliers
    ])


def load_and_split() -> Dict:
    """
    Load Iris, engineer features, oversample, split into
    train / val / test sets. Returns a tidy dict of splits.
    """
    log.info("Loading Iris dataset …")
    raw = load_iris()
    X, y = raw.data, raw.target
    cfg  = CFG.data

    log.info(f"Raw shape: {X.shape}  |  Classes: {raw.target_names.tolist()}")

    # ── hold-out test set (stratified)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size    = cfg.test_size,
        random_state = cfg.random_state,
        stratify     = y if cfg.stratify else None,
    )

    # ── validation split from remaining
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size    = val_frac,
        random_state = cfg.random_state,
        stratify     = y_tmp if cfg.stratify else None,
    )

    # ── fit preprocessing on train only
    prep = build_preprocessing_pipeline()
    X_train_p = prep.fit_transform(X_train)
    X_val_p   = prep.transform(X_val)
    X_test_p  = prep.transform(X_test)

    log.info(f"After feature engineering: {X_train_p.shape[1]} features")

    # ── oversampling (train only, to avoid leakage)
    if cfg.augment_minority:
        X_train_p, y_train = smote_oversample(X_train_p, y_train, cfg.random_state)
        log.info(f"After SMOTE oversampling: {X_train_p.shape[0]} train samples")

    splits = dict(
        X_train=X_train_p, y_train=y_train,
        X_val=X_val_p,     y_val=y_val,
        X_test=X_test_p,   y_test=y_test,
        preprocessor=prep,
        feature_names=prep.named_steps["engineer"].get_feature_names_out(raw.feature_names),
        class_names=raw.target_names,
        raw=raw,
    )
    log.info("Data pipeline complete.")
    return splits

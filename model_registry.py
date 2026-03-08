"""
src/utils/model_registry.py
─────────────────────────────────────────────────────────────────
Lightweight MLOps model registry:
  • Save / load models with joblib
  • Version-stamped metadata (mirrors MLflow / BentoML patterns)
  • Leaderboard tracking
"""

from __future__ import annotations
import os, json
import joblib
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

log = get_logger("ModelRegistry")


class ModelRegistry:
    def __init__(self, artifact_dir: str):
        self.model_dir  = os.path.join(artifact_dir, "models")
        self.meta_path  = os.path.join(self.model_dir, "registry.json")
        os.makedirs(self.model_dir, exist_ok=True)
        self._registry: List[Dict] = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self._registry = json.load(f)

    def register(self, name: str, model: Any, metrics: Dict,
                 params: Optional[Dict] = None, tags: Optional[Dict] = None):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.replace(' ','_')}_{ts}.joblib"
        path     = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)

        entry = dict(
            name=name, path=path, timestamp=ts,
            metrics={k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float, np.floating))},
            params=params or {}, tags=tags or {}
        )
        self._registry.append(entry)
        self._save()
        log.info(f"[Registry] Registered '{name}'  acc={metrics.get('accuracy',0):.4f}")
        return entry

    def load(self, name: str, version: int = -1) -> Any:
        matches = [e for e in self._registry if e["name"] == name]
        if not matches:
            raise KeyError(f"No model named '{name}' in registry")
        return joblib.load(matches[version]["path"])

    def leaderboard(self, metric: str = "accuracy") -> List[Dict]:
        valid = [e for e in self._registry if metric in e["metrics"]]
        return sorted(valid, key=lambda e: e["metrics"][metric], reverse=True)

    def print_leaderboard(self, metric: str = "accuracy", top_k: int = 10):
        board = self.leaderboard(metric)[:top_k]
        header = f"\n{'Rank':<5} {'Model':<28} {metric.upper():>10}  {'F1':>8}  {'MCC':>8}  {'Kappa':>8}"
        print("=" * 75)
        print("  MODEL LEADERBOARD")
        print("=" * 75)
        print(header)
        print("-" * 75)
        for i, e in enumerate(board, 1):
            m = e["metrics"]
            print(f"{i:<5} {e['name']:<28} "
                  f"{m.get(metric, 0):>10.4f}  "
                  f"{m.get('f1_weighted', 0):>8.4f}  "
                  f"{m.get('mcc', 0):>8.4f}  "
                  f"{m.get('kappa', 0):>8.4f}")
        print("=" * 75)

    def _save(self):
        with open(self.meta_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

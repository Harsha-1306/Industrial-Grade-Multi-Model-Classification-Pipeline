"""
main.py
─────────────────────────────────────────────────────────────────
Industrial ML Pipeline Orchestrator
────────────────────────────────────
Stages:
  1. Data ingestion & feature engineering (sklearn pipelines + SMOTE)
  2. Hyperparameter optimisation (RandomizedSearchCV)
  3. Custom Deep Neural Network (NumPy/SciPy — PyTorch-style API)
  4. Ensemble layer: Stacking + Soft-Voting
  5. Comprehensive evaluation (accuracy, F1, MCC, Kappa, AUC, log-loss,
     bootstrap CI, calibration)
  6. Explainability (permutation importance, PDP, entropy analysis)
  7. Model Registry (MLOps — save / version / leaderboard)
  8. 4-Dashboard visualisation suite

Architecture mirrors patterns used in production ML systems.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

from configs.config import CFG, ARTIFACT_DIR, LOG_DIR
from src.utils.logger import get_logger
from src.utils.model_registry import ModelRegistry
from src.data.pipeline import load_and_split
from src.models.neural_net import DeepNeuralNetwork
from src.models.ensemble import (
    get_base_estimators, StackingEnsemble, SoftVotingEnsemble
)
from src.training.hpo import run_hpo
from src.evaluation.evaluator import evaluate_model
from src.explainability.explainer import (
    permutation_importance, partial_dependence, prediction_confidence_breakdown
)
from src.evaluation.visualizer import (
    plot_data_dashboard, plot_model_comparison,
    plot_explainability, plot_nn_training
)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "plots"),   exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "models"),  exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "reports"), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger("Main", LOG_DIR, CFG.log_level)
registry = ModelRegistry(ARTIFACT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str):
    w = 70
    log.info("─" * w)
    log.info(f"  {text}")
    log.info("─" * w)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("  INDUSTRIAL ML PIPELINE  |  scikit-learn + NumPy Neural Net")
    log.info(f"  Experiment: {CFG.name}")
    log.info("=" * 70)

    # ── STAGE 1: Data ────────────────────────────────────────────────────────
    banner("STAGE 1 — Data Ingestion & Feature Engineering")
    splits = load_and_split()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    class_names      = splits["class_names"]
    feature_names    = splits["feature_names"]
    n_features       = X_train.shape[1]
    n_classes        = len(class_names)

    log.info(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── STAGE 2: HPO ─────────────────────────────────────────────────────────
    banner("STAGE 2 — Hyperparameter Optimisation (RandomizedSearchCV)")
    hpo_results = run_hpo(X_train, y_train)

    # ── STAGE 3: Neural Network ──────────────────────────────────────────────
    banner("STAGE 3 — Custom Deep Neural Network (NumPy backend)")
    nn = DeepNeuralNetwork(CFG.neural_net, n_features, n_classes)
    nn.fit(X_train, y_train, X_val, y_val)
    nn_score = nn.score(X_test, y_test)
    log.info(f"[NeuralNet] Test Accuracy = {nn_score:.4f}")

    # ── STAGE 4: Ensemble ────────────────────────────────────────────────────
    banner("STAGE 4 — Ensemble (Stacking + Soft-Voting)")
    base_ests = {n: r["best_estimator"] for n, r in hpo_results.items()}

    # Stacking
    stacker = StackingEnsemble(
        base_models  = base_ests,
        n_folds      = CFG.hpo.cv_folds,
        seed         = CFG.seed
    )
    stacker.fit(X_train, y_train)

    # Soft-Voting (reuse HPO best estimators, already fitted)
    voter = SoftVotingEnsemble(base_ests)
    voter.fit(X_train, y_train)

    # ── STAGE 5: Evaluation ──────────────────────────────────────────────────
    banner("STAGE 5 — Comprehensive Evaluation")
    eval_results = []

    # Individual HPO-tuned models
    for name, res in hpo_results.items():
        er = evaluate_model(name, res["best_estimator"],
                            X_test, y_test, class_names)
        eval_results.append(er)
        registry.register(name, res["best_estimator"],
                          er["metrics"], params=res["best_params"])

    # Neural Network
    nn_eval = evaluate_model("DeepNeuralNet", nn, X_test, y_test, class_names)
    eval_results.append(nn_eval)
    registry.register("DeepNeuralNet", nn, nn_eval["metrics"])

    # Stacking
    stack_eval = evaluate_model("StackingEnsemble", stacker,
                                X_test, y_test, class_names)
    eval_results.append(stack_eval)
    registry.register("StackingEnsemble", stacker, stack_eval["metrics"])

    # SoftVoting
    vote_eval = evaluate_model("SoftVotingEnsemble", voter,
                               X_test, y_test, class_names)
    eval_results.append(vote_eval)
    registry.register("SoftVotingEnsemble", voter, vote_eval["metrics"])

    registry.print_leaderboard("accuracy")

    # ── STAGE 6: Explainability ──────────────────────────────────────────────
    banner("STAGE 6 — Explainability Analysis")
    best_eval = max(eval_results, key=lambda r: r["metrics"]["accuracy"])
    best_name = best_eval["name"]
    log.info(f"Running explainability on best model: {best_name}")

    best_model = registry.load(best_name)
    perm_imp   = permutation_importance(
        best_model, X_test, y_test, feature_names, n_repeats=50
    )

    pdp_list = []
    for fi in range(min(4, n_features)):
        pdp = partial_dependence(best_model, X_test, fi)
        pdp["feature_name"] = feature_names[fi]
        pdp["class_names"]  = class_names
        pdp_list.append(pdp)

    conf_bd = prediction_confidence_breakdown(best_model, X_test, class_names)

    # ── STAGE 7: Visualisation ───────────────────────────────────────────────
    banner("STAGE 7 — Generating Visualisation Dashboards")
    plot_dir = os.path.join(ARTIFACT_DIR, "plots")

    p1 = plot_data_dashboard(splits, plot_dir)
    log.info(f"Dashboard 1 saved → {p1}")

    p2 = plot_model_comparison(eval_results, splits, plot_dir)
    log.info(f"Dashboard 2 saved → {p2}")

    p3 = plot_explainability(perm_imp, pdp_list, conf_bd, plot_dir)
    log.info(f"Dashboard 3 saved → {p3}")

    p4 = plot_nn_training(nn, nn_eval, splits, plot_dir)
    log.info(f"Dashboard 4 saved → {p4}")

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(f"  PIPELINE COMPLETE  |  Total time: {elapsed:.1f}s")
    log.info(f"  Best model: {best_name}  "
             f"acc={best_eval['metrics']['accuracy']:.4f}  "
             f"f1={best_eval['metrics']['f1_weighted']:.4f}  "
             f"mcc={best_eval['metrics']['mcc']:.4f}")
    log.info("=" * 70)

    return dict(
        eval_results=eval_results,
        best_model=best_name,
        plots=[p1, p2, p3, p4],
    )


if __name__ == "__main__":
    main()

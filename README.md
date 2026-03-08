# 🧠 Industrial-Grade Multi-Model Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-backend-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)

> A fully modular, production-grade machine learning pipeline that covers the complete lifecycle of a classification problem — from raw data ingestion to explainability, ensemble learning, and a versioned MLOps model registry. Engineered to mirror real-world industry standards.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Model Results](#-model-results)
- [Dashboards](#-dashboards)
- [Adapting to Your Own Dataset](#-adapting-to-your-own-dataset)
- [MLOps: Model Registry](#-mlops-model-registry)
- [Explainability](#-explainability)
- [CV / Resume Summary](#-cv--resume-summary)

---

## 🎯 Project Overview

This project is **not just a model training script** — it is a full end-to-end ML system built with the same architectural principles used by data science teams at production companies. It demonstrates how to go from raw tabular data to a deployed, versioned, and explainable classification system.

**Main Aim:** To automate and standardize the full ML lifecycle:
1. Ingest and validate raw data
2. Engineer features using custom sklearn-compatible transformers
3. Handle class imbalance via SMOTE-style oversampling
4. Tune hyperparameters automatically across multiple algorithms
5. Train a custom deep neural network (PyTorch-style API, NumPy backend)
6. Build a 2-level stacking ensemble and soft-voting ensemble
7. Evaluate all models with industrial-grade metrics and bootstrap confidence intervals
8. Explain predictions using model-agnostic explainability techniques
9. Register, version, and track all models in a lightweight MLOps registry
10. Produce a 4-dashboard visualisation suite

Although trained on the Iris dataset by default, **every module is fully dataset-agnostic** and can be applied to any binary or multi-class tabular classification problem (medical, finance, automotive, HR, cybersecurity, etc.).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔧 Custom Transformers | sklearn-compatible `FeatureEngineer` and `OutlierClipper` transformers |
| ⚖️ SMOTE Oversampling | Custom minority class interpolation — no external dependency |
| 🤖 Deep Neural Network | Scratch-built DNN: Adam, BatchNorm, Dropout, Cosine LR, Early Stopping |
| 🔍 HPO | RandomizedSearchCV across RF, GBM, SVM, LR with stratified CV |
| 🏗️ Stacking Ensemble | 2-level OOF stacking — zero data leakage |
| 🗳️ Soft Voting | Weighted probability averaging across tuned base models |
| 📊 Comprehensive Metrics | Accuracy, F1, MCC, Kappa, AUC, Log-Loss, Bootstrap CI |
| 🔬 Explainability | Permutation importance, PDP, prediction entropy analysis |
| 🗂️ Model Registry | MLflow-inspired versioned model storage with JSON metadata |
| 📁 Structured Logging | File + console logging with timestamps across all modules |
| 📈 4 Dashboards | Data, Model Comparison, Explainability, Neural Network deep-dive |

---

## 📁 Project Structure

```
ml_industrial/
│
├── main.py                          # Pipeline orchestrator — run this
│
├── configs/
│   └── config.py                    # Central config hub (dataclasses)
│
├── src/
│   ├── data/
│   │   └── pipeline.py              # Data loading, feature engineering, SMOTE, splitting
│   │
│   ├── models/
│   │   ├── neural_net.py            # Custom DNN (NumPy backend, PyTorch-style API)
│   │   └── ensemble.py              # Stacking + Soft-Voting ensembles
│   │
│   ├── training/
│   │   └── hpo.py                   # Hyperparameter optimisation (RandomizedSearchCV)
│   │
│   ├── evaluation/
│   │   ├── evaluator.py             # Metrics: Acc, F1, MCC, Kappa, AUC, Bootstrap CI
│   │   └── visualizer.py            # 4-dashboard matplotlib visualisation suite
│   │
│   ├── explainability/
│   │   └── explainer.py             # Permutation importance, PDP, entropy analysis
│   │
│   └── utils/
│       ├── logger.py                # Structured file + console logger
│       └── model_registry.py        # MLOps: versioned model storage + leaderboard
│
├── artifacts/
│   ├── models/                      # Saved .joblib model files + registry.json
│   ├── plots/                       # Generated dashboard PNGs
│   └── reports/                     # Reserved for future HTML/PDF reports
│
├── logs/                            # Per-run timestamped log files
│
└── README.md                        # This file
```

---

## 🏗️ Pipeline Architecture

```
Raw Data (any CSV / built-in dataset)
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 1: Data Pipeline           │
│  • Imputation (median)            │
│  • Outlier clipping (IQR)         │
│  • Feature engineering            │
│    (ratios, areas, interactions)  │
│  • RobustScaler normalization     │
│  • SMOTE oversampling             │
│  • Stratified train/val/test      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 2: Hyperparameter          │
│  Optimisation (HPO)               │
│  • RandomizedSearchCV             │
│  • Models: RF, GBM, SVM, LR      │
│  • Stratified K-Fold CV           │
│  • Scoring: F1-Weighted           │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 3: Deep Neural Network     │
│  • Configurable hidden layers     │
│  • Batch Normalisation            │
│  • Dropout regularisation         │
│  • Adam optimiser                 │
│  • Cosine LR schedule             │
│  • Early stopping + best restore  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 4: Ensemble Layer          │
│  • 2-Level Stacking (OOF)         │
│  • Soft-Voting Ensemble           │
│  • Meta-learner: Logistic Reg.    │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 5: Evaluation              │
│  • Accuracy + Bootstrap CI        │
│  • F1, Precision, Recall          │
│  • MCC, Cohen's Kappa             │
│  • ROC-AUC (OvR), Log-Loss        │
│  • Calibration analysis           │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 6: Explainability          │
│  • Permutation importance         │
│  • Partial Dependence Plots       │
│  • Prediction entropy             │
│  • Per-sample confidence          │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  STAGE 7: Visualisation &         │
│  MLOps Registry                   │
│  • 4 professional dashboards      │
│  • Versioned model saving         │
│  • JSON metadata + leaderboard    │
│  • Structured log files           │
└───────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Core Language
- **Python 3.9+**

### Machine Learning
- **scikit-learn** — Pipelines, transformers, HPO, base classifiers, calibration, metrics
- **NumPy** — Custom neural network backend, numerical operations
- **SciPy** — Statistical distributions for hyperparameter search spaces

### Algorithms
| Algorithm | Library | Role |
|---|---|---|
| Random Forest | scikit-learn | Base model + HPO |
| Gradient Boosting | scikit-learn | Base model + HPO |
| Support Vector Machine | scikit-learn | Base model + HPO |
| Logistic Regression | scikit-learn | Base model + Meta-learner |
| Deep Neural Network | NumPy (custom) | Stage 3 model |
| Stacking Ensemble | Custom | Stage 4 |
| Soft-Voting Ensemble | Custom | Stage 4 |

### MLOps & Engineering
- **joblib** — Model serialisation
- **logging** — Structured file + console logging
- **dataclasses** — Type-safe centralised configuration
- **json** — Model registry metadata storage

### Visualisation
- **matplotlib** — All 4 dashboards (dark-themed, production-grade)

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourname/ml-industrial-pipeline.git
cd ml-industrial-pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install numpy scipy scikit-learn matplotlib pandas joblib
```

**Requirements summary:**
```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
matplotlib>=3.7
pandas>=2.0
joblib>=1.2
```

---

## ▶️ Usage

### Run the full pipeline
```bash
python main.py
```

### What happens when you run it:
```
Stage 1  →  Loads data, engineers 9 features, applies SMOTE oversampling
Stage 2  →  Runs HPO across 4 classifiers (12 iterations each, 3-fold CV)
Stage 3  →  Trains custom deep neural network (80 epochs, early stopping)
Stage 4  →  Builds stacking + soft-voting ensembles
Stage 5  →  Evaluates all 7 models, prints leaderboard
Stage 6  →  Runs explainability (50-repeat permutation importance + PDP)
Stage 7  →  Generates 4 dashboards, saves to artifacts/plots/
```

### Expected output
```
======================================================================
  INDUSTRIAL ML PIPELINE  |  scikit-learn + NumPy Neural Net
  Experiment: iris_industrial_v1
======================================================================
...
  MODEL LEADERBOARD
======================================================================
Rank  Model                          ACCURACY        F1       MCC     Kappa
----------------------------------------------------------------------
1     RandomForest                     0.9667    0.9667    0.9516    0.9500
2     StackingEnsemble                 0.9667    0.9667    0.9516    0.9500
3     SoftVotingEnsemble               0.9667    0.9667    0.9516    0.9500
...
======================================================================
  PIPELINE COMPLETE  |  Total time: ~48s
```

---

## ⚙️ Configuration

All settings live in `configs/config.py`. Key parameters:

```python
# Neural Network
NeuralNetConfig(
    hidden_layers  = [128, 64, 32],   # Layer sizes
    activation     = "relu",           # relu | leaky_relu | tanh
    dropout_rate   = 0.30,
    learning_rate  = 1e-3,
    lr_schedule    = "cosine",         # cosine | step | constant
    epochs         = 200,
    patience       = 20,               # Early stopping
    use_batch_norm = True,
)

# Hyperparameter Optimisation
HPOConfig(
    n_iter    = 12,                    # Search iterations per model
    cv_folds  = 3,                     # Cross-validation folds
    scoring   = "f1_weighted",
)

# Data
DataConfig(
    test_size        = 0.20,
    val_size         = 0.15,
    augment_minority = True,           # SMOTE oversampling
)
```

---

## 📊 Model Results

| Model | Accuracy | F1 (Weighted) | MCC | Kappa |
|---|---|---|---|---|
| Random Forest | **0.9667** | **0.9667** | **0.9516** | **0.9500** |
| Stacking Ensemble | **0.9667** | **0.9667** | **0.9516** | **0.9500** |
| Soft-Voting Ensemble | **0.9667** | **0.9667** | **0.9516** | **0.9500** |
| Gradient Boosting | 0.9333 | 0.9327 | 0.9061 | 0.9000 |
| SVM | 0.9333 | 0.9333 | 0.9000 | 0.9000 |
| Logistic Regression | 0.9333 | 0.9333 | 0.9000 | 0.9000 |
| Deep Neural Network | 0.9333 | 0.9327 | 0.9061 | 0.9000 |

All results include **95% bootstrap confidence intervals**.

---

## 🖼️ Dashboards

The pipeline auto-generates 4 professional dark-themed dashboards:

| Dashboard | Contents |
|---|---|
| `01_data_dashboard.png` | Feature scatter plots, violin plots, correlation heatmap, PCA projection, engineered feature distributions, class balance |
| `02_model_comparison.png` | Accuracy bar chart with CI, multi-metric grouped bars, ROC-AUC, confusion matrix, calibration curves, log-loss |
| `03_explainability.png` | Permutation importance, Partial Dependence Plots, prediction entropy histogram, per-sample confidence scatter |
| `04_neural_network.png` | Train/val loss curves, weight heatmap, confusion matrix, ROC curves, probability distributions |

---

## 🔄 Adapting to Your Own Dataset

To apply this pipeline to any other domain (medical, automotive, finance, HR, etc.):

**Step 1 — Replace the data loader** in `src/data/pipeline.py`:
```python
# Replace this:
raw = load_iris()
X, y = raw.data, raw.target

# With your own data:
import pandas as pd
df = pd.read_csv("your_data.csv")
X  = df.drop("target_column", axis=1).values
y  = df["target_column"].values
```

**Step 2 — Update feature names** in the same file:
```python
feature_names = list(df.drop("target_column", axis=1).columns)
class_names   = df["target_column"].unique()
```

**Step 3 — Customize feature engineering** in `FeatureEngineer.transform()`:
```python
# Add domain-specific features relevant to your problem
# e.g. for medical: BMI = weight / height^2
# e.g. for finance: debt_to_income = debt / income
```

**Step 4 — Run:**
```bash
python main.py
```

Everything else — HPO, neural net, ensembles, evaluation, explainability, registry — adapts automatically.

---

## 🗂️ MLOps: Model Registry

Every trained model is automatically versioned and saved:

```
artifacts/models/
├── RandomForest_20260307_190538.joblib
├── StackingEnsemble_20260307_190539.joblib
├── DeepNeuralNet_20260307_190538.joblib
└── registry.json                          ← metadata + leaderboard
```

**Load any saved model:**
```python
from src.utils.model_registry import ModelRegistry

registry = ModelRegistry("artifacts/")
model    = registry.load("RandomForest")        # loads latest version
model    = registry.load("RandomForest", version=0)  # loads specific version

# Print leaderboard
registry.print_leaderboard(metric="accuracy")
```

---

## 🔬 Explainability

The pipeline uses **model-agnostic** explainability — works on any model without special dependencies:

```python
from src.explainability.explainer import permutation_importance, partial_dependence

# Permutation importance — how much accuracy drops when each feature is shuffled
perm = permutation_importance(model, X_test, y_test, feature_names, n_repeats=50)

# Partial Dependence Plot — how average prediction changes with one feature
pdp = partial_dependence(model, X_test, feature_idx=2)
```

**Top features identified (Iris dataset):**
1. `petal_area` — engineered feature (petal_l × petal_w) — Δ acc = 0.061
2. `petal length` — Δ acc = 0.030
3. `petal width` — Δ acc = 0.011

---

## 📄 CV / Resume Summary

> **Industrial-Grade ML Classification Pipeline** | Python, scikit-learn, NumPy
>
> Designed and implemented a modular, production-ready machine learning pipeline in Python following industry software engineering standards. Built a custom deep neural network from scratch using NumPy with a PyTorch-inspired API, incorporating Adam optimisation, Batch Normalisation, Dropout, cosine LR scheduling, and early stopping. Implemented automated hyperparameter optimisation using RandomizedSearchCV, a 2-level stacking ensemble with out-of-fold blending, and a soft-voting ensemble. Developed model-agnostic explainability using permutation importance and Partial Dependence Plots. Designed a lightweight MLOps model registry for versioned artifact storage and leaderboard tracking. Produced a 4-dashboard professional visualisation suite.
>
> **Tech:** Python · NumPy · scikit-learn · SciPy · joblib · matplotlib · Adam Optimiser · Batch Normalisation · Stacking Ensemble · RandomizedSearchCV · Bootstrap CI · MCC · Cohen's Kappa · ROC-AUC · Permutation Importance · PDP · MLOps

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 👤 Author

Built as a demonstration of industrial ML engineering practices.
Designed to be adapted, extended, and used as a portfolio project or production template.

---

*⭐ If this project helped you, consider giving it a star!*

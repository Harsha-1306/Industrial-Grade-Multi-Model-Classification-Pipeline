"""
configs/config.py
─────────────────────────────────────────────────────────────────
Central configuration hub — mirrors industry practice of keeping
all hyper-parameters, paths, and experiment settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
LOG_DIR      = os.path.join(BASE_DIR, "logs")

@dataclass
class DataConfig:
    test_size:        float = 0.20
    val_size:         float = 0.15
    random_state:     int   = 42
    stratify:         bool  = True
    augment_minority: bool  = True   # SMOTE-style oversampling

@dataclass
class NeuralNetConfig:
    """
    Config for our from-scratch Neural Network (NumPy/SciPy backend).
    Mirrors PyTorch/TF model configs used in industry.
    """
    hidden_layers:    List[int] = field(default_factory=lambda: [128, 64, 32])
    activation:       str  = "relu"          # relu | leaky_relu | tanh
    dropout_rate:     float = 0.30
    learning_rate:    float = 1e-3
    lr_schedule:      str  = "cosine"        # cosine | step | constant
    weight_decay:     float = 1e-4           # L2 regularisation
    batch_size:       int  = 32
    epochs:           int  = 80
    patience:         int  = 15              # early stopping
    optimizer:        str  = "adam"          # adam | sgd_momentum
    use_batch_norm:   bool = True

@dataclass
class EnsembleConfig:
    voting:           str  = "soft"
    use_stacking:     bool = True
    meta_learner:     str  = "logistic"

@dataclass
class HPOConfig:
    """Hyperparameter optimisation (grid + random search)."""
    strategy:         str  = "random"        # grid | random
    n_iter:           int  = 12
    cv_folds:         int  = 3
    scoring:          str  = "f1_weighted"
    n_jobs:           int  = 1

@dataclass
class ExperimentConfig:
    name:             str  = "iris_industrial_v1"
    data:             DataConfig       = field(default_factory=DataConfig)
    neural_net:       NeuralNetConfig  = field(default_factory=NeuralNetConfig)
    ensemble:         EnsembleConfig   = field(default_factory=EnsembleConfig)
    hpo:              HPOConfig        = field(default_factory=HPOConfig)
    seed:             int  = 42
    log_level:        str  = "INFO"
    save_artifacts:   bool = True
    LOG_DIR:          str  = field(default_factory=lambda: LOG_DIR)

CFG = ExperimentConfig()

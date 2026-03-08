"""
src/models/neural_net.py
─────────────────────────────────────────────────────────────────
Production-grade Neural Network built from scratch in NumPy.
Architecture mirrors PyTorch / TensorFlow designs:
  • Configurable hidden layers
  • Batch Normalisation
  • Dropout regularisation
  • Adam / SGD-Momentum optimisers
  • Cosine / Step learning-rate schedules
  • Early stopping with best-weight restore
  • sklearn-compatible API (fit / predict / predict_proba)
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import NeuralNetConfig
from src.utils.logger import get_logger

log = get_logger("NeuralNet")


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────
def relu(x):         return np.maximum(0, x)
def relu_grad(x):    return (x > 0).astype(float)
def leaky_relu(x, a=0.01):  return np.where(x > 0, x, a * x)
def leaky_relu_grad(x, a=0.01): return np.where(x > 0, 1.0, a)
def tanh_act(x):     return np.tanh(x)
def tanh_grad(x):    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_one_hot):
    return -np.mean(np.sum(y_one_hot * np.log(probs + 1e-12), axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedules
# ─────────────────────────────────────────────────────────────────────────────
def cosine_lr(base_lr, epoch, total_epochs, warmup=10):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def step_lr(base_lr, epoch, step=50, gamma=0.5):
    return base_lr * (gamma ** (epoch // step))


# ─────────────────────────────────────────────────────────────────────────────
# Batch Norm layer
# ─────────────────────────────────────────────────────────────────────────────
class BatchNorm:
    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma    = np.ones(n_features)
        self.beta     = np.zeros(n_features)
        self.momentum = momentum
        self.eps      = eps
        self.running_mean = np.zeros(n_features)
        self.running_var  = np.ones(n_features)
        # cache for backward
        self._cache: dict = {}

    def forward(self, x, training=True):
        if training:
            mu  = x.mean(axis=0)
            var = x.var(axis=0)
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*mu
            self.running_var  = (1 - self.momentum)*self.running_var  + self.momentum*var
            self._cache = dict(x=x, x_hat=x_hat, mu=mu, var=var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        x, x_hat, mu, var = (self._cache[k] for k in ("x","x_hat","mu","var"))
        N = x.shape[0]
        dgamma = (dout * x_hat).sum(axis=0)
        dbeta  = dout.sum(axis=0)
        dx_hat = dout * self.gamma
        dvar   = (-0.5 * dx_hat * (x - mu) * (var + self.eps)**(-1.5)).sum(axis=0)
        dmu    = (-dx_hat / np.sqrt(var + self.eps)).sum(axis=0)
        dx     = dx_hat / np.sqrt(var + self.eps) + 2*dvar*(x - mu)/N + dmu/N
        return dx, dgamma, dbeta


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
class DeepNeuralNetwork:
    """
    Multi-layer neural network with:
      BatchNorm · Dropout · Adam/SGD · Cosine/Step LR schedule
    sklearn-compatible: fit(), predict(), predict_proba(), score()
    """

    def __init__(self, cfg: NeuralNetConfig, n_features: int, n_classes: int):
        self.cfg        = cfg
        self.n_features = n_features
        self.n_classes  = n_classes
        self.training   = True
        self._init_params()
        self._init_optimiser()
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

    # ── weight initialisation (He for ReLU, Xavier for tanh) ─────────────────
    def _init_params(self):
        cfg   = self.cfg
        layer_dims = [self.n_features] + cfg.hidden_layers + [self.n_classes]
        rng   = np.random.default_rng(42)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        self.bn: List[Optional[BatchNorm]] = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            if cfg.activation in ("relu","leaky_relu"):
                scale = np.sqrt(2.0 / fan_in)          # He init
            else:
                scale = np.sqrt(1.0 / fan_in)           # Xavier
            self.W.append(rng.normal(0, scale, (fan_in, layer_dims[i+1])))
            self.b.append(np.zeros(layer_dims[i+1]))
            # BatchNorm on hidden layers only
            if cfg.use_batch_norm and i < len(layer_dims) - 2:
                self.bn.append(BatchNorm(layer_dims[i+1]))
            else:
                self.bn.append(None)

    def _init_optimiser(self):
        """Adam moment estimates."""
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
        self.m_gam = [np.zeros(bn.gamma.shape) if bn else None for bn in self.bn]
        self.v_gam = [np.zeros(bn.gamma.shape) if bn else None for bn in self.bn]
        self.m_bet = [np.zeros(bn.beta.shape)  if bn else None for bn in self.bn]
        self.v_bet = [np.zeros(bn.beta.shape)  if bn else None for bn in self.bn]
        self._step = 0

    def _activate(self, x):
        act = self.cfg.activation
        if act == "relu":        return relu(x),        relu_grad
        if act == "leaky_relu":  return leaky_relu(x),  leaky_relu_grad
        return tanh_act(x), tanh_grad

    def _forward(self, X: np.ndarray) -> Tuple[List, List, np.ndarray]:
        cache_z, cache_a = [], [X]
        a = X
        for i, (W, b, bn) in enumerate(zip(self.W, self.b, self.bn)):
            z = a @ W + b
            if i < len(self.W) - 1:                    # hidden layers
                if bn:
                    z = bn.forward(z, self.training)
                a_new, _ = self._activate(z)
                # Dropout (inverted, train only)
                if self.training and self.cfg.dropout_rate > 0:
                    mask  = (np.random.rand(*a_new.shape) > self.cfg.dropout_rate)
                    a_new = a_new * mask / (1 - self.cfg.dropout_rate)
                    z     = z * mask  # keep mask aligned
            else:                                       # output layer
                a_new = softmax(z)
            cache_z.append(z)
            cache_a.append(a_new)
            a = a_new
        return cache_z, cache_a, a

    def _backward(self, cache_z, cache_a, y_oh: np.ndarray) -> Tuple[List, List]:
        N   = y_oh.shape[0]
        dW  = [None] * len(self.W)
        db  = [None] * len(self.b)
        d_bn_gam = [None] * len(self.bn)
        d_bn_bet = [None] * len(self.bn)

        # output layer gradient (softmax + cross-entropy combined)
        da = (cache_a[-1] - y_oh) / N       # ∂L/∂z_output

        for i in reversed(range(len(self.W))):
            a_prev = cache_a[i]             # activation from previous layer
            dW[i]  = a_prev.T @ da + self.cfg.weight_decay * self.W[i] / N
            db[i]  = da.sum(axis=0)
            da_prev = da @ self.W[i].T      # gradient w.r.t previous layer's output

            if i > 0:                       # hidden layer: backprop through BN + activation
                # BN sits between pre-activation z and activation
                if self.bn[i - 1] is not None:
                    da_prev, dgam, dbet = self.bn[i - 1].backward(da_prev)
                    d_bn_gam[i - 1] = dgam
                    d_bn_bet[i - 1] = dbet
                # activation gradient uses pre-activation z of layer i-1
                z_prev = cache_z[i - 1]
                _, grad_fn = self._activate(z_prev)
                da = da_prev * grad_fn(z_prev)
            else:
                da = da_prev

        return dW, db, d_bn_gam, d_bn_bet

    def _adam_update(self, grads_W, grads_b, grads_gam, grads_bet, lr):
        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = self._step
        for i in range(len(self.W)):
            for (m, v, g, param) in [
                (self.m_W, self.v_W, grads_W, self.W),
                (self.m_b, self.v_b, grads_b, self.b),
            ]:
                if g[i] is None: continue
                m[i] = b1 * m[i] + (1-b1) * g[i]
                v[i] = b2 * v[i] + (1-b2) * g[i]**2
                m_hat = m[i] / (1 - b1**t)
                v_hat = v[i] / (1 - b2**t)
                param[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # BN params
            if self.bn[i] and grads_gam[i] is not None:
                for (m, v, g, param) in [
                    (self.m_gam, self.v_gam, grads_gam, [bn.gamma if bn else None for bn in self.bn]),
                    (self.m_bet, self.v_bet, grads_bet, [bn.beta  if bn else None for bn in self.bn]),
                ]:
                    if g[i] is None or param[i] is None: continue
                    m[i] = b1 * m[i] + (1-b1) * g[i]
                    v[i] = b2 * v[i] + (1-b2) * g[i]**2
                    m_hat = m[i] / (1 - b1**t)
                    v_hat = v[i] / (1 - b2**t)
                    param[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _get_lr(self, epoch):
        base = self.cfg.learning_rate
        if self.cfg.lr_schedule == "cosine":
            return cosine_lr(base, epoch, self.cfg.epochs)
        if self.cfg.lr_schedule == "step":
            return step_lr(base, epoch)
        return base

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cfg  = self.cfg
        n_cls = self.n_classes
        rng   = np.random.default_rng(42)

        def to_onehot(y):
            oh = np.zeros((len(y), n_cls))
            oh[np.arange(len(y)), y] = 1
            return oh

        best_val_loss = np.inf
        best_weights  = None
        patience_cnt  = 0

        for epoch in range(cfg.epochs):
            self.training = True
            lr = self._get_lr(epoch)

            # ── mini-batch SGD
            idx = rng.permutation(len(X_train))
            for start in range(0, len(X_train), cfg.batch_size):
                bi  = idx[start:start+cfg.batch_size]
                xb, yb = X_train[bi], y_train[bi]
                yb_oh = to_onehot(yb)
                cz, ca, _ = self._forward(xb)
                gW, gb, ggam, gbet = self._backward(cz, ca, yb_oh)
                self._adam_update(gW, gb, ggam, gbet, lr)

            # ── epoch metrics
            self.training = False
            _, _, out_tr = self._forward(X_train)
            tr_loss = cross_entropy(out_tr, to_onehot(y_train))
            self.train_losses.append(tr_loss)

            if X_val is not None:
                _, _, out_v = self._forward(X_val)
                vl = cross_entropy(out_v, to_onehot(y_val))
                self.val_losses.append(vl)

                if vl < best_val_loss - 1e-5:
                    best_val_loss = vl
                    best_weights  = self._snapshot()
                    patience_cnt  = 0
                else:
                    patience_cnt += 1

                if patience_cnt >= cfg.patience:
                    log.info(f"[NeuralNet] Early stop @ epoch {epoch+1}  val_loss={vl:.4f}")
                    break

            if (epoch + 1) % 20 == 0:
                vl_str = f"  val={self.val_losses[-1]:.4f}" if X_val is not None else ""
                log.info(f"[NeuralNet] epoch {epoch+1:>4d}/{cfg.epochs}  "
                         f"lr={lr:.5f}  train={tr_loss:.4f}{vl_str}")

        if best_weights:
            self._restore(best_weights)
        self.training = False
        return self

    def _snapshot(self):
        return (
            [w.copy() for w in self.W],
            [b.copy() for b in self.b],
            [(bn.gamma.copy(), bn.beta.copy(),
              bn.running_mean.copy(), bn.running_var.copy()) if bn else None
             for bn in self.bn],
        )

    def _restore(self, snap):
        W, b, bns = snap
        for i in range(len(self.W)):
            self.W[i] = W[i]; self.b[i] = b[i]
            if bns[i] and self.bn[i]:
                g, be, rm, rv = bns[i]
                self.bn[i].gamma        = g
                self.bn[i].beta         = be
                self.bn[i].running_mean = rm
                self.bn[i].running_var  = rv

    def predict_proba(self, X):
        self.training = False
        _, _, probs = self._forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

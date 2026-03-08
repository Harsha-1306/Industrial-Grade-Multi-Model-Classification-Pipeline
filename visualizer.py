"""
src/evaluation/visualizer.py
─────────────────────────────────────────────────────────────────
Industrial-grade visualisation suite:
  Dashboard 1 – Data & Feature Engineering
  Dashboard 2 – Model Leaderboard & Calibration
  Dashboard 3 – Best Model Deep Dive
  Dashboard 4 – Explainability (Permutation Importance + PDP)
  Dashboard 5 – Neural Network Training Curves
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from typing import Dict, List, Any
import os

# ── Design system ─────────────────────────────────────────────────────────────
P = dict(
    bg="#0F1117", card="#1A1D27", accent="#7C3AED",
    c1="#7C3AED", c2="#06B6D4", c3="#10B981", c4="#F59E0B", c5="#EF4444",
    text="#F1F5F9", muted="#64748B", grid="#1E2433",
)
PALETTE = [P["c1"], P["c2"], P["c3"], P["c4"], P["c5"]]

def _fig(rows, cols, h, w=None):
    w = w or cols * 5.5
    fig = plt.figure(figsize=(w, h), facecolor=P["bg"])
    return fig

def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(P["card"])
    ax.tick_params(colors=P["muted"], labelsize=8)
    ax.xaxis.label.set_color(P["muted"])
    ax.yaxis.label.set_color(P["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(P["grid"])
    ax.grid(True, color=P["grid"], linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, color=P["text"], fontsize=10, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=P["muted"], fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=P["muted"], fontsize=8)


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 1: Data exploration
# ═════════════════════════════════════════════════════════════════════════════
def plot_data_dashboard(splits: Dict, out_dir: str):
    X_raw  = splits["raw"].data
    y      = splits["raw"].target
    feats  = splits["raw"].feature_names
    cnames = splits["raw"].target_names

    fig = _fig(2, 3, 10, 18)
    fig.suptitle("DATA & FEATURE ENGINEERING DASHBOARD",
                 color=P["text"], fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── Scatter matrix (first 2 features) ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _ax_style(ax, "Petal Length vs Width (Raw)")
    for i, (cls, col) in enumerate(zip(cnames, PALETTE)):
        mask = y == i
        ax.scatter(X_raw[mask, 2], X_raw[mask, 3], c=col, s=30, alpha=0.7, label=cls)
    ax.legend(fontsize=7, labelcolor=P["text"],
              facecolor=P["card"], edgecolor=P["grid"])
    ax.set_xlabel("Petal Length (cm)", color=P["muted"], fontsize=8)
    ax.set_ylabel("Petal Width (cm)",  color=P["muted"], fontsize=8)

    # ── Violin plots ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _ax_style(ax2, "Feature Distributions (violin)")
    parts = ax2.violinplot([X_raw[:, i] for i in range(4)],
                           positions=range(4), showmedians=True)
    for body, col in zip(parts["bodies"], PALETTE):
        body.set_facecolor(col); body.set_alpha(0.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["Sep.L","Sep.W","Pet.L","Pet.W"],
                        color=P["muted"], fontsize=8)

    # ── Correlation heatmap ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _ax_style(ax3, "Feature Correlation")
    corr = np.corrcoef(X_raw.T)
    im   = ax3.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
    labels = ["Sep.L","Sep.W","Pet.L","Pet.W"]
    ax3.set_xticklabels(labels, color=P["muted"], fontsize=7, rotation=30)
    ax3.set_yticklabels(labels, color=P["muted"], fontsize=7)
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     fontsize=8, color="black" if abs(corr[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ── PCA 2D projection ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _ax_style(ax4, "PCA 2-D Projection")
    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(X_raw)
    for i, (cls, col) in enumerate(zip(cnames, PALETTE)):
        mask = y == i
        ax4.scatter(Xp[mask, 0], Xp[mask, 1], c=col, s=30, alpha=0.7, label=cls)
    ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                   color=P["muted"], fontsize=8)
    ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                   color=P["muted"], fontsize=8)
    ax4.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── Engineered features distribution ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _ax_style(ax5, "Engineered: Petal Area by Class")
    X_eng = splits["X_train"]
    y_tr  = splits["y_train"]
    for i, (cls, col) in enumerate(zip(cnames, PALETTE)):
        vals = X_eng[y_tr == i, 7]  # petal_area
        ax5.hist(vals, bins=20, alpha=0.6, color=col, label=cls)
    ax5.set_xlabel("Petal Area (petal_l × petal_w)", color=P["muted"], fontsize=8)
    ax5.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── Class balance bar ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _ax_style(ax6, "Class Balance (after SMOTE)")
    classes, cnts = np.unique(splits["y_train"], return_counts=True)
    ax6.bar([cnames[c] for c in classes], cnts, color=PALETTE[:len(classes)], alpha=0.8)
    for xi, cnt in enumerate(cnts):
        ax6.text(xi, cnt + 0.5, str(cnt), ha="center", color=P["text"], fontsize=9)
    ax6.set_ylabel("Samples", color=P["muted"], fontsize=8)
    ax6.tick_params(colors=P["muted"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "01_data_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2: Model leaderboard & calibration
# ═════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(eval_results: List[Dict], splits: Dict, out_dir: str):
    names   = [r["name"] for r in eval_results]
    accs    = [r["metrics"]["accuracy"]    for r in eval_results]
    f1s     = [r["metrics"]["f1_weighted"] for r in eval_results]
    mccs    = [r["metrics"]["mcc"]         for r in eval_results]
    kappas  = [r["metrics"]["kappa"]       for r in eval_results]
    ci_lo   = [r["metrics"]["acc_ci_lo"]   for r in eval_results]
    ci_hi   = [r["metrics"]["acc_ci_hi"]   for r in eval_results]

    fig = _fig(2, 3, 10, 18)
    fig.suptitle("MODEL COMPARISON & CALIBRATION DASHBOARD",
                 color=P["text"], fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Accuracy bar with CI ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _ax_style(ax, "Accuracy (95% Bootstrap CI)")
    x   = np.arange(len(names))
    bars = ax.bar(x, accs, color=PALETTE[:len(names)], alpha=0.85, zorder=3)
    yerr = np.array([np.array(accs) - np.array(ci_lo),
                     np.array(ci_hi) - np.array(accs)])
    ax.errorbar(x, accs, yerr=yerr, fmt="none", color=P["text"],
                capsize=4, linewidth=1.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in names], rotation=30, ha="right",
                       color=P["muted"], fontsize=7)
    ax.set_ylim(min(accs)-0.05, 1.02)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", color=P["text"], fontsize=7)

    # ── Multi-metric spider / grouped bar ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _ax_style(ax2, "Multi-Metric Comparison")
    w = 0.2
    xi = np.arange(len(names))
    for mi, (vals, label, col) in enumerate(
        [(accs,"Accuracy",P["c1"]),(f1s,"F1",P["c2"]),
         (mccs,"MCC",P["c3"]),(kappas,"Kappa",P["c4"])]
    ):
        ax2.bar(xi + mi*w - 1.5*w, vals, w, label=label, color=col, alpha=0.8)
    ax2.set_xticks(xi)
    ax2.set_xticklabels([n[:8] for n in names], rotation=30, ha="right",
                        color=P["muted"], fontsize=7)
    ax2.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── ROC AUC per model ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _ax_style(ax3, "ROC AUC (macro OvR)")
    aucs_val = [r["metrics"].get("roc_auc_ovr", 0) for r in eval_results]
    bars3 = ax3.barh(names, aucs_val, color=PALETTE[:len(names)], alpha=0.8)
    ax3.set_xlim(0.8, 1.01)
    for bar, v in zip(bars3, aucs_val):
        ax3.text(v + 0.001, bar.get_y()+bar.get_height()/2,
                 f"{v:.4f}", va="center", color=P["text"], fontsize=7)
    ax3.tick_params(colors=P["muted"], labelsize=7)

    # ── Confusion matrix best ─────────────────────────────────────────────
    best_r = max(eval_results, key=lambda r: r["metrics"]["accuracy"])
    ax4    = fig.add_subplot(gs[1, 0])
    _ax_style(ax4, f"Confusion Matrix\n({best_r['name']})")
    cm   = best_r["cm"]
    cnames = splits["class_names"]
    im   = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
    ax4.set_xticklabels(cnames, rotation=30, ha="right", color=P["muted"], fontsize=8)
    ax4.set_yticklabels(cnames, color=P["muted"], fontsize=8)
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > thresh else P["text"],
                     fontsize=12, fontweight="bold")

    # ── Calibration curves ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _ax_style(ax5, "Calibration Curves")
    ax5.plot([0,1],[0,1], "--", color=P["muted"], linewidth=1, label="Perfect")
    X_t, y_t = splits["X_test"], splits["y_test"]
    for r, col in zip(eval_results[:4], PALETTE):
        if r["y_proba"] is None: continue
        frac_pos, mean_pred = calibration_curve(
            (y_t == 0).astype(int), r["y_proba"][:, 0], n_bins=8
        )
        ax5.plot(mean_pred, frac_pos, "o-", color=col, alpha=0.8,
                 linewidth=1.5, markersize=4, label=r["name"][:10])
    ax5.legend(fontsize=6, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── Log-loss comparison ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _ax_style(ax6, "Log-Loss Comparison")
    ll_vals = [r["metrics"].get("log_loss", np.nan) for r in eval_results]
    colors_ll = [P["c3"] if v == min([x for x in ll_vals if not np.isnan(x)])
                 else P["c5"] for v in ll_vals]
    bars6 = ax6.bar(range(len(names)), ll_vals, color=colors_ll, alpha=0.8)
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels([n[:10] for n in names], rotation=30, ha="right",
                        color=P["muted"], fontsize=7)
    ax6.set_ylabel("Log-Loss (lower=better)", color=P["muted"], fontsize=8)
    for bar, v in zip(bars6, ll_vals):
        if not np.isnan(v):
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f"{v:.3f}", ha="center", color=P["text"], fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "02_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 3: Explainability
# ═════════════════════════════════════════════════════════════════════════════
def plot_explainability(perm_imp: Dict, pdp_data: List[Dict],
                        conf_breakdown: Dict, out_dir: str):
    fig = _fig(2, 3, 10, 18)
    fig.suptitle("EXPLAINABILITY DASHBOARD",
                 color=P["text"], fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Permutation importance ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    _ax_style(ax, "Permutation Feature Importance (Δ Accuracy)")
    imp  = perm_imp["importances"]
    stds = perm_imp["stds"]
    feat = perm_imp["feature_names"]
    top  = min(12, len(feat))
    y_pos = np.arange(top)
    bars = ax.barh(y_pos, imp[:top],
                   xerr=stds[:top], color=PALETTE[0], alpha=0.8,
                   error_kw=dict(ecolor=P["muted"], capsize=3))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat[:top], color=P["muted"], fontsize=8)
    ax.set_xlabel("Mean Accuracy Drop (higher = more important)",
                  color=P["muted"], fontsize=8)
    for bar in bars:
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", color=P["text"], fontsize=7)

    # ── PDP plots ─────────────────────────────────────────────────────────
    for pi, pdp in enumerate(pdp_data[:2]):
        ax2 = fig.add_subplot(gs[0, 2] if pi == 0 else gs[1, 0])
        _ax_style(ax2, f"PDP: {pdp['feature_name']}")
        for ci, (cls, col) in enumerate(zip(pdp["class_names"], PALETTE)):
            ax2.plot(pdp["grid"], pdp["avg_proba"][:, ci], color=col,
                     linewidth=2, label=cls)
        ax2.set_xlabel(pdp["feature_name"], color=P["muted"], fontsize=8)
        ax2.set_ylabel("Avg Predicted Probability", color=P["muted"], fontsize=8)
        ax2.legend(fontsize=7, labelcolor=P["text"],
                   facecolor=P["card"], edgecolor=P["grid"])

    # ── Confidence entropy distribution ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _ax_style(ax3, "Prediction Confidence (Entropy)")
    ax3.hist(conf_breakdown["entropy"], bins=25, color=P["c2"], alpha=0.8)
    ax3.axvline(conf_breakdown["entropy"].mean(), color=P["c4"],
                linestyle="--", linewidth=1.5, label=f"Mean={conf_breakdown['entropy'].mean():.3f}")
    ax3.set_xlabel("Entropy (lower = more certain)", color=P["muted"], fontsize=8)
    ax3.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── Top-probability scatter ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _ax_style(ax4, "Per-Sample Top-Class Probability")
    cnames = conf_breakdown["class_names"]
    pred   = conf_breakdown["pred"]
    top_p  = conf_breakdown["top_p"]
    for ci, (cls, col) in enumerate(zip(cnames, PALETTE)):
        mask = pred == ci
        ax4.scatter(np.where(mask)[0], top_p[mask], c=col, s=20, alpha=0.7, label=cls)
    ax4.axhline(0.9, color=P["muted"], linestyle="--", linewidth=1)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Sample Index", color=P["muted"], fontsize=8)
    ax4.set_ylabel("Max Probability", color=P["muted"], fontsize=8)
    ax4.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "03_explainability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 4: Neural Network training curves
# ═════════════════════════════════════════════════════════════════════════════
def plot_nn_training(nn_model, eval_result: Dict, splits: Dict, out_dir: str):
    fig = _fig(2, 3, 10, 18)
    fig.suptitle("NEURAL NETWORK DEEP DIVE",
                 color=P["text"], fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Loss curves ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    _ax_style(ax, "Training & Validation Loss Curves")
    epochs = range(1, len(nn_model.train_losses)+1)
    ax.plot(epochs, nn_model.train_losses, color=P["c1"], linewidth=1.5, label="Train Loss")
    if nn_model.val_losses:
        ax.plot(range(1, len(nn_model.val_losses)+1),
                nn_model.val_losses, color=P["c3"], linewidth=1.5, label="Val Loss")
    ax.set_xlabel("Epoch", color=P["muted"], fontsize=8)
    ax.set_ylabel("Cross-Entropy Loss", color=P["muted"], fontsize=8)
    ax.legend(fontsize=8, labelcolor=P["text"],
              facecolor=P["card"], edgecolor=P["grid"])

    # ── Weight magnitude heatmap (first layer) ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _ax_style(ax2, "L1 Weight Matrix (Layer 1)")
    W0 = nn_model.W[0]
    im = ax2.imshow(W0.T, aspect="auto", cmap="RdBu_r")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xlabel("Input Features", color=P["muted"], fontsize=8)
    ax2.set_ylabel("Hidden Units",   color=P["muted"], fontsize=8)

    # ── NN Confusion matrix ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _ax_style(ax3, "NN Confusion Matrix (Test Set)")
    cm = eval_result["cm"]
    cnames = splits["class_names"]
    im3 = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks(range(3)); ax3.set_yticks(range(3))
    ax3.set_xticklabels(cnames, rotation=30, ha="right", color=P["muted"], fontsize=7)
    ax3.set_yticklabels(cnames, color=P["muted"], fontsize=7)
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j]>thresh else P["text"],
                     fontsize=12, fontweight="bold")

    # ── ROC (NN) ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _ax_style(ax4, "NN ROC Curves (OvR)")
    y_bin = label_binarize(splits["y_test"], classes=[0,1,2])
    y_p   = eval_result["y_proba"]
    ax4.plot([0,1],[0,1],"--",color=P["muted"],linewidth=1)
    for ci, (cls, col) in enumerate(zip(cnames, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:,ci], y_p[:,ci])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, color=col, linewidth=2,
                 label=f"{cls} AUC={roc_auc:.3f}")
    ax4.set_xlabel("FPR", color=P["muted"], fontsize=8)
    ax4.set_ylabel("TPR", color=P["muted"], fontsize=8)
    ax4.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    # ── Probability output distribution ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _ax_style(ax5, "NN Output Probability Distribution")
    for ci, (cls, col) in enumerate(zip(cnames, PALETTE)):
        ax5.hist(y_p[:, ci], bins=20, alpha=0.6, color=col, label=cls)
    ax5.set_xlabel("Predicted Probability", color=P["muted"], fontsize=8)
    ax5.legend(fontsize=7, labelcolor=P["text"],
               facecolor=P["card"], edgecolor=P["grid"])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "04_neural_network.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    return path

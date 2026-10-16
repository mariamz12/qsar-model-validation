"""Publication-style validation figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import ValidationResult

TRAIN = dict(color="#2b6cb0", label="Training")
TEST = dict(color="#dd6b20", label="Test")


def williams_plot(res: ValidationResult, path: str) -> None:
    """Leverage vs. standardized residuals with h* and ±3σ boundaries."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(res.lev_train, res.sres_train, s=28, alpha=0.8, edgecolor="white", **TRAIN)
    ax.scatter(res.lev_test, res.sres_test, s=36, marker="^", alpha=0.9, edgecolor="white", **TEST)
    ax.axvline(res.h_star, ls="--", color="grey", lw=1)
    for y in (-3, 3):
        ax.axhline(y, ls="--", color="grey", lw=1)
    ax.text(res.h_star, ax.get_ylim()[1], f" h* = {res.h_star:.3f}", va="top", fontsize=9, color="grey")
    ax.set_xlabel("Leverage (h)")
    ax.set_ylabel("Standardized residual")
    ax.set_title("Williams plot — applicability domain")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def predicted_vs_experimental(res: ValidationResult, y_train, y_test, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_train, res.y_train_pred, s=28, alpha=0.8, edgecolor="white", **TRAIN)
    ax.scatter(y_test, res.y_test_pred, s=36, marker="^", alpha=0.9, edgecolor="white", **TEST)
    lo = min(np.min(y_train), np.min(y_test), res.y_train_pred.min(), res.y_test_pred.min())
    hi = max(np.max(y_train), np.max(y_test), res.y_train_pred.max(), res.y_test_pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", lw=1)
    m = res.metrics
    ax.text(0.04, 0.96,
            f"R² train = {m['R2_train']:.3f}\nQ²LOO = {m['Q2_LOO']:.3f}\n"
            f"Q²F2 test = {m['Q2_F2']:.3f}\nCCC test = {m['CCC_test']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Experimental activity")
    ax.set_ylabel("Predicted activity")
    ax.set_title("Predicted vs. experimental")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def y_randomization_plot(res: ValidationResult, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(res.yrand[:, 0], res.yrand[:, 1], s=18, alpha=0.6, color="grey",
               label=f"Scrambled models (n={len(res.yrand)})")
    ax.scatter([res.metrics["R2_train"]], [res.metrics["Q2_LOO"]], s=90, marker="*",
               color="#c53030", label="Real model", zorder=3)
    ax.set_xlabel("R²")
    ax.set_ylabel("Q²LOO")
    ax.set_title("Y-randomization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

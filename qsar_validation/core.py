"""Core validation routines for linear (MLR) QSAR models.

Independent implementation of standard, published QSAR validation methods:

* Leverage-based applicability domain (Williams plot), following
  Gramatica (2007) QSAR Comb. Sci. 26, 694-701.
* Y-randomization (response permutation), following
  Rücker, Rücker & Meringer (2007) J. Chem. Inf. Model. 47, 2345-2357.
* External validation metrics Q2_F1, Q2_F2, Q2_F3 and the concordance
  correlation coefficient (CCC), following Chirico & Gramatica (2011)
  J. Chem. Inf. Model. 51, 2320-2335.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict


# ---------------------------------------------------------------------------
# Applicability domain
# ---------------------------------------------------------------------------

def _design_matrix(X: np.ndarray) -> np.ndarray:
    """Descriptor matrix with an intercept column, as used by MLR."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def leverages(X_train: np.ndarray, X_query: np.ndarray | None = None) -> np.ndarray:
    """Leverage h_i = x_i^T (X^T X)^-1 x_i for each query compound.

    The hat matrix is built from the training set only, so test compounds
    are judged against the chemical space the model was fitted on.
    """
    Xt = _design_matrix(X_train)
    Xq = Xt if X_query is None else _design_matrix(X_query)
    xtx_inv = np.linalg.pinv(Xt.T @ Xt)
    return np.einsum("ij,jk,ik->i", Xq, xtx_inv, Xq)


def warning_leverage(n_train: int, n_descriptors: int) -> float:
    """Critical leverage h* = 3(p + 1) / n."""
    return 3.0 * (n_descriptors + 1) / n_train


def standardized_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                           train_residual_sd: float) -> np.ndarray:
    """Residuals divided by the training-set residual standard deviation."""
    return (np.asarray(y_true) - np.asarray(y_pred)) / train_residual_sd


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def q2_loo(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out cross-validated Q2 on the training set."""
    pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return r2(y, pred)


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(y_true, float), np.asarray(y_pred, float)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def external_q2(y_train: np.ndarray, y_test: np.ndarray,
                y_test_pred: np.ndarray) -> dict[str, float]:
    """Q2_F1 (training mean), Q2_F2 (test mean) and Q2_F3 (per-compound) metrics."""
    y_train = np.asarray(y_train, float)
    y_test, y_test_pred = np.asarray(y_test, float), np.asarray(y_test_pred, float)
    press = np.sum((y_test - y_test_pred) ** 2)
    q2_f1 = 1 - press / np.sum((y_test - y_train.mean()) ** 2)
    q2_f2 = 1 - press / np.sum((y_test - y_test.mean()) ** 2)
    q2_f3 = 1 - (press / len(y_test)) / (np.sum((y_train - y_train.mean()) ** 2) / len(y_train))
    return {"Q2_F1": q2_f1, "Q2_F2": q2_f2, "Q2_F3": q2_f3}


# ---------------------------------------------------------------------------
# Y-randomization
# ---------------------------------------------------------------------------

def y_randomization(X: np.ndarray, y: np.ndarray, n_iter: int = 100,
                    seed: int = 0) -> np.ndarray:
    """Refit the model on shuffled responses.

    Returns an (n_iter, 2) array of [R2, Q2_LOO] for each scrambled model.
    A model capturing real structure-activity signal should clearly beat
    every scrambled one.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_iter, 2))
    for i in range(n_iter):
        y_perm = rng.permutation(y)
        model = LinearRegression().fit(X, y_perm)
        out[i] = r2(y_perm, model.predict(X)), q2_loo(X, y_perm)
    return out


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    metrics: dict[str, float]
    h_star: float
    lev_train: np.ndarray
    lev_test: np.ndarray
    sres_train: np.ndarray
    sres_test: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    yrand: np.ndarray


def validate(X_train, y_train, X_test, y_test, n_yrand: int = 100,
             seed: int = 0) -> ValidationResult:
    """Fit an MLR model and run the full validation workflow."""
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train, y_test = np.asarray(y_train, float), np.asarray(y_test, float)
    n, p = X_train.shape

    model = LinearRegression().fit(X_train, y_train)
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)

    resid = y_train - y_train_pred
    resid_sd = np.sqrt(np.sum(resid ** 2) / (n - p - 1))

    metrics = {
        "n_train": n, "n_test": len(y_test), "n_descriptors": p,
        "R2_train": r2(y_train, y_train_pred),
        "Q2_LOO": q2_loo(X_train, y_train),
        "RMSE_train": rmse(y_train, y_train_pred),
        "R2_test": r2(y_test, y_test_pred),
        "RMSE_test": rmse(y_test, y_test_pred),
        "CCC_test": ccc(y_test, y_test_pred),
        **external_q2(y_train, y_test, y_test_pred),
    }

    yrand = y_randomization(X_train, y_train, n_iter=n_yrand, seed=seed)
    metrics["yrand_max_R2"] = float(yrand[:, 0].max())
    metrics["yrand_max_Q2"] = float(yrand[:, 1].max())

    h_star = warning_leverage(n, p)
    lev_test = leverages(X_train, X_test)
    sres_test = standardized_residuals(y_test, y_test_pred, resid_sd)
    inside = (lev_test <= h_star) & (np.abs(sres_test) <= 3)
    metrics["test_inside_AD_pct"] = float(100 * inside.mean())
    if inside.sum() >= 3:
        # Same external metrics, restricted to test compounds inside the domain.
        q_in = external_q2(y_train, y_test[inside], y_test_pred[inside])
        metrics["Q2_F2_inside_AD"] = q_in["Q2_F2"]
        metrics["CCC_test_inside_AD"] = ccc(y_test[inside], y_test_pred[inside])

    return ValidationResult(
        metrics=metrics, h_star=h_star,
        lev_train=leverages(X_train), lev_test=lev_test,
        sres_train=standardized_residuals(y_train, y_train_pred, resid_sd),
        sres_test=sres_test,
        y_train_pred=y_train_pred, y_test_pred=y_test_pred, yrand=yrand,
    )

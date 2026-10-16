import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from qsar_validation import (ccc, external_q2, leverages, q2_loo, r2, validate,
                             warning_leverage, y_randomization)


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(0, 0.2, 40)
    return X[:30], y[:30], X[30:], y[30:]


def test_training_leverages_sum_to_p_plus_1(data):
    X_train, *_ = data
    # The trace of the hat matrix equals the number of fitted parameters.
    assert leverages(X_train).sum() == pytest.approx(X_train.shape[1] + 1)


def test_leverage_matches_explicit_hat_matrix(data):
    X_train, _, X_test, _ = data
    Xd = np.column_stack([np.ones(len(X_train)), X_train])
    Xq = np.column_stack([np.ones(len(X_test)), X_test])
    expected = np.diag(Xq @ np.linalg.inv(Xd.T @ Xd) @ Xq.T)
    assert leverages(X_train, X_test) == pytest.approx(expected)


def test_warning_leverage():
    assert warning_leverage(n_train=30, n_descriptors=3) == pytest.approx(0.4)


def test_r2_matches_sklearn(data):
    X, y, *_ = data
    pred = LinearRegression().fit(X, y).predict(X)
    assert r2(y, pred) == pytest.approx(r2_score(y, pred))


def test_q2_loo_matches_manual_loop(data):
    X, y, *_ = data
    preds = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        preds.append(LinearRegression().fit(X[mask], y[mask]).predict(X[i:i + 1])[0])
    assert q2_loo(X, y) == pytest.approx(r2_score(y, preds))


def test_ccc_perfect_and_shifted():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ccc(y, y) == pytest.approx(1.0)
    assert ccc(y, y + 1.0) < 1.0


def test_external_q2_perfect_prediction():
    m = external_q2(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5]), np.array([1.5, 2.5]))
    assert all(v == pytest.approx(1.0) for v in m.values())


def test_y_randomization_destroys_signal(data):
    X, y, *_ = data
    scrambled = y_randomization(X, y, n_iter=50, seed=0)
    assert scrambled.shape == (50, 2)
    assert scrambled[:, 1].max() < q2_loo(X, y)


def test_validate_end_to_end(data):
    res = validate(*data, n_yrand=20)
    m = res.metrics
    assert m["R2_train"] > 0.9 and m["Q2_LOO"] > 0.9 and m["Q2_F2"] > 0.8
    assert m["yrand_max_Q2"] < m["Q2_LOO"]
    assert len(res.lev_test) == len(data[3])

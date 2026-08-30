"""Tests for metis.evaluation.metrics (I5)."""
import numpy as np
import pytest

from metis.evaluation.metrics import (
    mae,
    nrmse,
    pointwise_metrics,
    r2,
    relative_l2,
    rmse,
)


@pytest.fixture
def y():
    rng = np.random.default_rng(0)
    return rng.normal(size=(30, 4))


def test_perfect_prediction(y):
    m = pointwise_metrics(y, y)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["relative_l2"] == 0
    assert m["r2"] == pytest.approx(1.0)


def test_mae_and_rmse_known_values():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, -1.0, 1.0])
    assert mae(a, b) == pytest.approx(1.0)
    assert rmse(a, b) == pytest.approx(1.0)


def test_nrmse_and_relative_l2_are_scale_invariant(y):
    pred = y + 0.1 * np.random.default_rng(1).normal(size=y.shape)
    assert nrmse(y, pred) == pytest.approx(nrmse(1000 * y, 1000 * pred))
    assert relative_l2(y, pred) == pytest.approx(relative_l2(1000 * y, 1000 * pred))


def test_r2_is_zero_for_mean_prediction(y):
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-9)


def test_r2_negative_for_worse_than_mean(y):
    assert r2(y, -y) < 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.zeros(3), np.zeros(4))

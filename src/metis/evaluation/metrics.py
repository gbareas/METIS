"""Generic pointwise prediction metrics (milestone I5).

Array-in, scalar-out. `pointwise_metrics` bundles the standard set;
`nrmse` and `relative_l2` are normalised by the true signal so they're
comparable across cases and variables of different magnitude.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-30


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(y_true, y_pred) -> float:
    """RMSE normalised by the true signal's RMS."""
    a, b = _pair(y_true, y_pred)
    denom = float(np.sqrt(np.mean(a**2)))
    return rmse(a, b) / (denom + _EPS)


def relative_l2(y_true, y_pred) -> float:
    """‖pred − true‖₂ / ‖true‖₂ (Frobenius over all elements)."""
    a, b = _pair(y_true, y_pred)
    return float(np.linalg.norm((b - a).ravel()) / (np.linalg.norm(a.ravel()) + _EPS))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (1 − SS_res / SS_tot). Can be
    negative for a prediction worse than the mean."""
    a, b = _pair(y_true, y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + _EPS)


def pointwise_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "relative_l2": relative_l2(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

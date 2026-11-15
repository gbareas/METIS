"""Feature scalers (milestone R4).

`StandardScaler` matches `metis.evaluation.regime.standardize`'s
convention exactly (population std, zero-variance column left at zero) so
it can replace that inline standardization once leakage-safe splits
matter.
"""
from __future__ import annotations

import numpy as np

from metis.data.preprocessing.base import Transform


class StandardScaler(Transform):
    """Column-wise z-score: ``(X - mean) / std``, fitted on the rows of
    `X` passed to `fit` (pass training rows only). A zero-variance column
    is scaled by 1, i.e. left mean-centred at zero."""

    def __init__(self, *, ddof: int = 0):
        self.ddof = ddof
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, X) -> StandardScaler:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=self.ddof)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        self._check_fitted()
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, X):
        self._check_fitted()
        return np.asarray(X, dtype=float) * self.std_ + self.mean_

    def get_params(self) -> dict:
        return {
            "ddof": self.ddof,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict) -> StandardScaler:
        obj = cls(ddof=params.get("ddof", 0))
        if params.get("mean") is not None:
            obj.mean_ = np.asarray(params["mean"], dtype=float)
            obj.std_ = np.asarray(params["std"], dtype=float)
        return obj


class MinMaxScaler(Transform):
    """Linear rescale of each column to `feature_range` (default [0, 1]),
    fitted on the rows passed to `fit`. A constant column maps to the low
    end of the range."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = tuple(feature_range)
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.min_ is not None

    def fit(self, X) -> MinMaxScaler:
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def _span(self) -> np.ndarray:
        span = self.max_ - self.min_
        return np.where(span > 0, span, 1.0)

    def transform(self, X):
        self._check_fitted()
        lo, hi = self.feature_range
        unit = (np.asarray(X, dtype=float) - self.min_) / self._span()
        return unit * (hi - lo) + lo

    def inverse_transform(self, X):
        self._check_fitted()
        lo, hi = self.feature_range
        unit = (np.asarray(X, dtype=float) - lo) / (hi - lo)
        return unit * self._span() + self.min_

    def get_params(self) -> dict:
        return {
            "feature_range": list(self.feature_range),
            "min": None if self.min_ is None else self.min_.tolist(),
            "max": None if self.max_ is None else self.max_.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict) -> MinMaxScaler:
        obj = cls(feature_range=tuple(params.get("feature_range", (0.0, 1.0))))
        if params.get("min") is not None:
            obj.min_ = np.asarray(params["min"], dtype=float)
            obj.max_ = np.asarray(params["max"], dtype=float)
        return obj

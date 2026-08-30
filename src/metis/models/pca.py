"""Linear PCA representation — the baseline every nonlinear model is
judged against (milestone I2 / I4).

Numpy/scipy only; no torch. `transform` gives PC scores, `inverse_transform`
reconstructs from them. Matches `metis.evaluation.regime.pca`'s SVD
convention.
"""
from __future__ import annotations

import numpy as np

from metis.models.base import RepresentationModel


class PCARepresentation(RepresentationModel):
    def __init__(self, latent_dim: int = 2, *, whiten: bool = False):
        self.latent_dim = int(latent_dim)
        self.whiten = bool(whiten)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None          # (latent_dim, n_features)
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.components_ is not None

    def fit(self, X) -> PCARepresentation:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _u, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = min(self.latent_dim, vt.shape[0])
        self.components_ = vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ratio_ = ((s**2) / np.sum(s**2))[:k]
        return self

    def transform(self, X):
        self._check_fitted()
        scores = (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T
        if self.whiten:
            scores = scores / self.singular_values_
        return scores

    def inverse_transform(self, Z):
        self._check_fitted()
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z * self.singular_values_
        return Z @ self.components_ + self.mean_

    def get_params(self) -> dict:
        arr = lambda a: None if a is None else a.tolist()
        return {
            "latent_dim": self.latent_dim,
            "whiten": self.whiten,
            "mean": arr(self.mean_),
            "components": arr(self.components_),
            "singular_values": arr(self.singular_values_),
            "explained_variance_ratio": arr(self.explained_variance_ratio_),
        }

    @classmethod
    def from_params(cls, params: dict) -> PCARepresentation:
        obj = cls(latent_dim=params["latent_dim"], whiten=params.get("whiten", False))
        if params.get("components") is not None:
            obj.mean_ = np.asarray(params["mean"], dtype=float)
            obj.components_ = np.asarray(params["components"], dtype=float)
            obj.singular_values_ = np.asarray(params["singular_values"], dtype=float)
            obj.explained_variance_ratio_ = np.asarray(
                params["explained_variance_ratio"], dtype=float
            )
        return obj

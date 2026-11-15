"""Representation-model interface (milestone I4).

A `RepresentationModel` maps `X (n, n_features)` to a low-dimensional
latent `Z (n, latent_dim)` (`transform`) and back to a reconstruction
(`inverse_transform`). It reuses the preprocessing `Transform` contract —
`fit` / `transform` / `inverse_transform` / `get_params` / `from_params` /
`is_fitted`, plus `save` / `load` via the shared registry — so a linear
PCA baseline and a torch autoencoder are drop-in swappable.

Deliberately minimal: no sklearn-style `predict`, no generic pipeline.
"""
from __future__ import annotations

import numpy as np

from metis.data.preprocessing.base import Transform


class RepresentationModel(Transform):
    latent_dim: int

    def reconstruction_mse(self, X) -> float:
        """Mean squared reconstruction error over all elements of `X`."""
        X = np.asarray(X, dtype=float)
        X_hat = np.asarray(self.inverse_transform(self.transform(X)), dtype=float)
        return float(np.mean((X - X_hat) ** 2))

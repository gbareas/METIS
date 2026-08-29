"""Snapshot POD (method of snapshots) on a `SliceCase`'s fluctuation
fields.

Same algorithm and 0.99 energy-threshold convention as the already-
published `pub4_grassmann_rom/src/pod_slice.py`, generalized to operate on
`SliceCase` (in-memory, from `SliceReader`) instead of loading `.npy`
files by path — cross-checked against its cached results for
case01/s2_max_u/u (r=29, energy_captured=0.9900003143300564).

SPOD (the frequency-resolved variant) is out of scope: it needs the
physical timestep size, which isn't recorded alongside the slice
snapshots (only solver iteration numbers are).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metis.data.ingestion.slice_reader import SliceCase

DEFAULT_ENERGY_THRESHOLD = 0.99


@dataclass
class PODResult:
    modes: np.ndarray  # (N_dof, r) spatial modes Phi, orthonormal columns
    singular_values: np.ndarray  # (r,) truncated singular values
    singular_values_all: np.ndarray  # (n_snapshots,) full spectrum
    coefficients: np.ndarray  # (r, n_snapshots) temporal coefficients
    energy_captured: float
    energy_threshold: float
    shape: tuple[int, int]  # (NX, NZ) spatial shape, to unflatten a mode

    def mode_field(self, i: int) -> np.ndarray:
        """Mode `i` reshaped back to the (NX, NZ) spatial plane."""
        return self.modes[:, i].reshape(self.shape)

    def energy_fractions(self) -> np.ndarray:
        """Per-mode energy fraction over the *full* singular spectrum —
        the 'POD energy spectrum (leading-mode energy fractions)'
        modal-space feature from research_protocol.md."""
        return self.singular_values_all**2 / np.sum(self.singular_values_all**2)


def compute_pod(
    X_flat: np.ndarray, energy_threshold: float = DEFAULT_ENERGY_THRESHOLD
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Method-of-snapshots POD, identical algorithm to
    `pod_slice.py:compute_pod`. `X_flat`: (n_snapshots, n_dof) float64."""
    C = X_flat @ X_flat.T  # (n_snapshots, n_snapshots)
    eigvals, V = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals[::-1], 0.0)
    V = V[:, ::-1].copy()

    cumulative = np.cumsum(eigvals) / eigvals.sum()
    r = int(np.searchsorted(cumulative, energy_threshold)) + 1
    r = min(r, X_flat.shape[0])
    energy = float(cumulative[r - 1])

    sigma_all = np.sqrt(eigvals)
    sigma = sigma_all[:r]
    V_r = V[:, :r]

    modes = (X_flat.T @ V_r) / sigma[np.newaxis, :]  # (n_dof, r)
    coeffs = sigma[:, np.newaxis] * V_r.T  # (r, n_snapshots)

    return modes, sigma, sigma_all, coeffs, r, energy


def pod(
    case: SliceCase, field: str, energy_threshold: float = DEFAULT_ENERGY_THRESHOLD
) -> PODResult:
    """Snapshot POD of `field`'s fluctuation snapshots in `case`."""
    snaps = case.snapshots[field]  # (n_snapshots, NX, NZ)
    n_snap, nx, nz = snaps.shape
    X_flat = snaps.reshape(n_snap, nx * nz).astype(np.float64)

    modes, sigma, sigma_all, coeffs, _r, energy = compute_pod(X_flat, energy_threshold)
    return PODResult(
        modes=modes,
        singular_values=sigma,
        singular_values_all=sigma_all,
        coefficients=coeffs,
        energy_captured=energy,
        energy_threshold=energy_threshold,
        shape=(nx, nz),
    )

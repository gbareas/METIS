"""Wavenumber energy spectra along the homogeneous (periodic) in-plane
directions of an XZ DNS slice.

x and z are statistically homogeneous in this channel geometry, so a 1D
power spectral density is well-defined by FFT along either axis. Spectra
are ensemble-averaged over all available snapshots and spatially averaged
over the other in-plane direction — both act as independent realizations
of the same statistics, which is what makes a single time-resolved slice
(rather than a full 3D time series) enough for a usable spectrum estimate.

Only spatial (wavenumber) spectra are implemented — temporal/frequency
spectra would need the physical timestep size, which isn't recorded
alongside the slice snapshots (only solver iteration numbers are).
"""
from __future__ import annotations

import numpy as np

from metis.data.ingestion.slice_reader import SliceCase


def _one_sided_psd(x: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD along the last axis of `x`, averaged over all leading axes.

    Normalized so that `np.sum(psd) * (k[1] - k[0])` equals `np.mean(x**2)`
    (Parseval — the mean square, not the variance: the k=0/DC bin already
    carries the `mean(x)**2` term) — `psd` is an energy density, not a raw
    FFT magnitude.
    """
    n = x.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    coeffs = np.fft.rfft(x.astype(np.float64), axis=-1) / n
    power = np.abs(coeffs) ** 2
    power[..., 1:] *= 2.0
    if n % 2 == 0:
        power[..., -1] /= 2.0  # Nyquist bin is real-valued, not doubled
    psd = power.reshape(-1, power.shape[-1]).mean(axis=0)
    dk = k[1] - k[0]
    return k, psd / dk


def wavenumber_spectrum(case: SliceCase, field: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """1D energy spectrum E(k) of `field`'s fluctuation snapshots along the
    homogeneous direction `axis` ('x' or 'z'), for one `SliceCase`.

    Returns (k, E) with `np.sum(E) * (k[1] - k[0])` equal to the field's
    ensemble-and-plane-averaged mean square (Parseval check).
    """
    if axis not in ("x", "z"):
        raise ValueError(f"axis must be 'x' or 'z', got {axis!r}")
    snaps = case.snapshots[field]  # (n_snapshots, NX, NZ)
    axis_dim = 1 if axis == "x" else 2
    x = np.moveaxis(snaps, axis_dim, -1)  # (..., N_along_axis)
    dx = float(np.diff(case.coordinates[axis]).mean())
    return _one_sided_psd(x, dx)


def premultiplied_spectrum(case: SliceCase, field: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """k*E(k) — flattens the inertial-range slope for easier visual
    comparison across cases; standard practice for wall-turbulence spectra."""
    k, e = wavenumber_spectrum(case, field, axis)
    return k, k * e

"""Physical-diagnostic agreement between a reference field and a
prediction (milestone I5).

Low pointwise error does not mean a prediction is physically right — it
can still miss the mean/RMS profile shape or the energy spectrum. These
helpers quantify that, on plain arrays (no `DNSCase` / torch dependency):
a caller passes true/pred fields, or already-computed POD energy
fractions.
"""
from __future__ import annotations

import numpy as np

from metis.evaluation.metrics import pointwise_metrics, relative_l2
from metis.features.spectra import _one_sided_psd

_EPS = 1e-30


def _axis_profile(arr: np.ndarray, keep_axis: int, reducer) -> np.ndarray:
    """Reduce every axis except `keep_axis` -> 1-D profile."""
    moved = np.moveaxis(np.asarray(arr, dtype=float), keep_axis, 0)
    return reducer(moved.reshape(moved.shape[0], -1), axis=1)


def profile_agreement(true, pred, *, axis: int) -> dict:
    """Relative-L2 agreement of the mean and RMS (std) profiles along
    `axis` (all other axes averaged out)."""
    t_mean = _axis_profile(true, axis, np.mean)
    p_mean = _axis_profile(pred, axis, np.mean)
    t_rms = _axis_profile(true, axis, np.std)
    p_rms = _axis_profile(pred, axis, np.std)
    return {
        "mean_profile_rel_l2": relative_l2(t_mean, p_mean),
        "rms_profile_rel_l2": relative_l2(t_rms, p_rms),
    }


def spectrum_agreement(true, pred, *, axis: int, dx: float = 1.0) -> dict:
    """Agreement of the 1-D wavenumber spectrum along `axis`: relative L2
    of E(k), Pearson correlation of log E(k), and the fractional shift of
    the premultiplied-spectrum peak wavenumber."""
    t = np.moveaxis(np.asarray(true, dtype=float), axis, -1)
    p = np.moveaxis(np.asarray(pred, dtype=float), axis, -1)
    k, e_t = _one_sided_psd(t, dx)
    _, e_p = _one_sided_psd(p, dx)

    log_corr = float(np.corrcoef(np.log(e_t + _EPS), np.log(e_p + _EPS))[0, 1])
    kpeak_t = float(k[np.argmax(k * e_t)])
    kpeak_p = float(k[np.argmax(k * e_p)])
    return {
        "spectrum_rel_l2": relative_l2(e_t, e_p),
        "spectrum_log_corr": log_corr,
        "peak_wavenumber_true": kpeak_t,
        "peak_wavenumber_pred": kpeak_p,
        "peak_wavenumber_rel_shift": abs(kpeak_p - kpeak_t) / (kpeak_t + _EPS),
    }


def pod_energy_agreement(true_fractions, pred_fractions, *, k: int = 5) -> dict:
    """Agreement of the POD energy spectrum (per-mode energy fractions).
    Pass `PODResult.energy_fractions()` for each field."""
    a = np.asarray(true_fractions, dtype=float)
    b = np.asarray(pred_fractions, dtype=float)
    m = min(k, a.size, b.size)
    return {
        "energy_fraction_l1": float(np.sum(np.abs(a[:m] - b[:m]))),
        "leading_mode_energy_true": float(a[0]),
        "leading_mode_energy_pred": float(b[0]),
        "cumulative_energy_delta_topk": float(np.sum(a[:m]) - np.sum(b[:m])),
    }


def physical_field_report(
    true, pred, *, profile_axis: int, spectrum_axis: int | None = None, dx: float = 1.0
) -> dict:
    """Bundle: pointwise metrics + mean/RMS profile agreement (+ spectrum
    agreement if `spectrum_axis` is given)."""
    report = {
        "pointwise": pointwise_metrics(true, pred),
        **profile_agreement(true, pred, axis=profile_axis),
    }
    if spectrum_axis is not None:
        report.update(spectrum_agreement(true, pred, axis=spectrum_axis, dx=dx))
    return report

"""Tests for metis.evaluation.physical (I5)."""
import numpy as np
import pytest

from metis.evaluation.physical import (
    physical_field_report,
    pod_energy_agreement,
    profile_agreement,
    spectrum_agreement,
)


@pytest.fixture
def field():
    # (n_snapshots, H=profile axis, W=homogeneous axis)
    rng = np.random.default_rng(0)
    prof = np.tanh(np.linspace(-2, 2, 24))[None, :, None]
    return prof + 0.2 * rng.normal(size=(40, 24, 32))


def test_profile_agreement_zero_for_identical(field):
    a = profile_agreement(field, field, axis=1)
    assert a["mean_profile_rel_l2"] == pytest.approx(0.0, abs=1e-12)
    assert a["rms_profile_rel_l2"] == pytest.approx(0.0, abs=1e-12)


def test_profile_agreement_detects_a_scaled_rms(field):
    a = profile_agreement(field, 3.0 * field, axis=1)
    assert a["mean_profile_rel_l2"] == pytest.approx(2.0, rel=1e-6)   # (3-1)
    assert a["rms_profile_rel_l2"] == pytest.approx(2.0, rel=1e-6)


def test_spectrum_agreement_identical_field(field):
    s = spectrum_agreement(field, field, axis=2)
    assert s["spectrum_rel_l2"] == pytest.approx(0.0, abs=1e-12)
    assert s["spectrum_log_corr"] == pytest.approx(1.0)
    assert s["peak_wavenumber_rel_shift"] == pytest.approx(0.0)


def test_spectrum_agreement_worse_for_smoothed_field(field):
    from scipy.ndimage import uniform_filter1d

    smooth = uniform_filter1d(field, size=5, axis=2)
    s = spectrum_agreement(field, smooth, axis=2)
    # a box filter strips high-k energy: log-spectrum decorrelates and the
    # premultiplied-spectrum peak shifts to lower wavenumber
    assert s["spectrum_log_corr"] < 0.95
    assert s["peak_wavenumber_pred"] < s["peak_wavenumber_true"]
    assert s["peak_wavenumber_rel_shift"] > 0.1


def test_pod_energy_agreement():
    t = np.array([0.6, 0.25, 0.1, 0.03, 0.02])
    p = np.array([0.5, 0.3, 0.12, 0.05, 0.03])
    a = pod_energy_agreement(t, p, k=5)
    assert a["energy_fraction_l1"] == pytest.approx(0.1 + 0.05 + 0.02 + 0.02 + 0.01)
    assert a["leading_mode_energy_true"] == 0.6


def test_physical_field_report_bundles(field):
    r = physical_field_report(field, field * 1.01, profile_axis=1, spectrum_axis=2)
    assert set(r["pointwise"]) == {"mae", "rmse", "nrmse", "relative_l2", "r2"}
    assert "mean_profile_rel_l2" in r and "spectrum_log_corr" in r

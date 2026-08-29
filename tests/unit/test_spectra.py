import numpy as np
import pytest

from metis.data.ingestion.slice_reader import SliceReader
from metis.features.spectra import premultiplied_spectrum, wavenumber_spectrum
from metis.testing.mock_slices import generate


@pytest.fixture
def mock_slice_case(tmp_path):
    slice_dir = generate(tmp_path / "case01" / "s3_center", nx=32, nz=16, n_snapshots=50, seed=3)
    return SliceReader().read(slice_dir)


@pytest.mark.parametrize("axis", ["x", "z"])
def test_spectrum_satisfies_parseval(mock_slice_case, axis):
    k, e = wavenumber_spectrum(mock_slice_case, "u", axis)
    dk = k[1] - k[0]
    # Parseval relates the PSD to mean(x**2), not var(x) — the k=0 (DC) bin
    # already carries the mean(x)**2 term, so subtracting the mean again
    # (as .var() does) would double-count it.
    mean_square = float(np.mean(mock_slice_case.snapshots["u"].astype(np.float64) ** 2))
    assert np.sum(e) * dk == pytest.approx(mean_square, rel=1e-10)


def test_spectrum_wavenumbers_start_at_zero_and_increase(mock_slice_case):
    k, _ = wavenumber_spectrum(mock_slice_case, "T", "x")
    assert k[0] == 0.0
    assert np.all(np.diff(k) > 0)


def test_spectrum_shape_matches_nyquist_bins(mock_slice_case):
    k, e = wavenumber_spectrum(mock_slice_case, "cp", "z")
    nz = mock_slice_case.coordinates["z"].shape[0]
    assert k.shape == (nz // 2 + 1,)
    assert e.shape == k.shape


def test_premultiplied_spectrum_is_k_times_e(mock_slice_case):
    k, e = wavenumber_spectrum(mock_slice_case, "u", "x")
    k2, ke = premultiplied_spectrum(mock_slice_case, "u", "x")
    np.testing.assert_allclose(k, k2)
    np.testing.assert_allclose(ke, k * e)


def test_invalid_axis_raises(mock_slice_case):
    with pytest.raises(ValueError, match="axis must be"):
        wavenumber_spectrum(mock_slice_case, "u", "y")

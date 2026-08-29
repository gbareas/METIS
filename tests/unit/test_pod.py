import numpy as np
import pytest

from metis.data.ingestion.slice_reader import SliceReader
from metis.features.pod import pod
from metis.testing.mock_slices import generate


@pytest.fixture
def mock_slice_case(tmp_path):
    slice_dir = generate(tmp_path / "case01" / "s2_max_u", nx=8, nz=8, n_snapshots=40, seed=4)
    return SliceReader().read(slice_dir)


def test_modes_are_orthonormal(mock_slice_case):
    result = pod(mock_slice_case, "u", energy_threshold=0.99)
    gram = result.modes.T @ result.modes
    np.testing.assert_allclose(gram, np.eye(result.modes.shape[1]), atol=1e-8)


def test_full_rank_reconstructs_snapshots_exactly(mock_slice_case):
    n_snap, nx, nz = mock_slice_case.snapshots["u"].shape
    result = pod(mock_slice_case, "u", energy_threshold=1.0)
    assert result.singular_values.shape[0] == n_snap
    reconstructed = (result.modes @ result.coefficients).T  # (n_snap, n_dof)
    original = mock_slice_case.snapshots["u"].reshape(n_snap, nx * nz).astype(np.float64)
    np.testing.assert_allclose(reconstructed, original, atol=1e-8)


def test_lower_threshold_gives_fewer_or_equal_modes(mock_slice_case):
    loose = pod(mock_slice_case, "T", energy_threshold=0.5)
    tight = pod(mock_slice_case, "T", energy_threshold=0.99)
    full = pod(mock_slice_case, "T", energy_threshold=1.0)
    assert loose.singular_values.shape[0] <= tight.singular_values.shape[0]
    assert tight.singular_values.shape[0] <= full.singular_values.shape[0]
    assert loose.energy_captured <= tight.energy_captured + 1e-12


def test_energy_fractions_sum_to_one_and_are_nonincreasing(mock_slice_case):
    result = pod(mock_slice_case, "cp", energy_threshold=0.99)
    fractions = result.energy_fractions()
    assert fractions.shape == (mock_slice_case.snapshots["cp"].shape[0],)
    assert fractions.sum() == pytest.approx(1.0, rel=1e-10)
    assert np.all(np.diff(fractions) <= 1e-12)


def test_mode_field_reshapes_to_spatial_grid(mock_slice_case):
    result = pod(mock_slice_case, "u", energy_threshold=0.99)
    field0 = result.mode_field(0)
    assert field0.shape == result.shape
    np.testing.assert_allclose(field0.ravel(), result.modes[:, 0])

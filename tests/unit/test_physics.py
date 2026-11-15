import numpy as np
import pytest

from metis.data.ingestion.hdf5_reader import HDF5Reader
from metis.features.physics import (
    BULK_FIELDS,
    bulk_quantities,
    case_physics_summary,
    friction_reynolds,
    wall_normal_profiles,
    wall_quantities,
)
from metis.testing.mock_dns import generate


@pytest.fixture
def mock_case(tmp_path):
    h5_path, metadata_path = generate(tmp_path / "case_mock.h5", nx=6, ny=10, nz=6, seed=2)
    return HDF5Reader().read(h5_path, metadata_path=metadata_path)


def test_wall_normal_profiles_matches_manual_xz_average(mock_case):
    profiles = wall_normal_profiles(mock_case, ("avg_T",))
    expected = mock_case.fields["avg_T"][1:-1, :, 1:-1].mean(axis=(0, 2))
    assert profiles["avg_T"].shape == (mock_case.metadata.grid["Ny"] + 2,)
    np.testing.assert_allclose(profiles["avg_T"], expected)


def test_wall_quantities_symmetric_definition(mock_case):
    profiles = wall_normal_profiles(mock_case, BULK_FIELDS)
    y = mock_case.coordinates["y"]
    cw = wall_quantities(profiles, y, "cw")
    hw = wall_quantities(profiles, y, "hw")
    assert cw["T_w"] == pytest.approx(0.5 * (profiles["avg_T"][0] + profiles["avg_T"][1]))
    assert hw["T_w"] == pytest.approx(0.5 * (profiles["avg_T"][-1] + profiles["avg_T"][-2]))
    assert cw["dudy_w"] >= 0
    assert hw["dudy_w"] >= 0


def test_bulk_quantities_internal_consistency(mock_case):
    profiles = wall_normal_profiles(mock_case, BULK_FIELDS)
    bulk = bulk_quantities(profiles)
    assert bulk["Re_b"] == pytest.approx(
        bulk["rho_b"] * bulk["U_b"] * 175e-6 / bulk["mu_b"]
    )
    assert bulk["Pr_b"] == pytest.approx(bulk["mu_b"] * bulk["cp_b"] / bulk["kappa_b"])
    assert bulk["Br_b"] == pytest.approx(bulk["Pr_b"] * bulk["Ec_b"], rel=1e-6)
    assert bulk["Ma_b"] == pytest.approx(bulk["U_b"] / bulk["c_b"])


def test_friction_reynolds_positive(mock_case):
    profiles = wall_normal_profiles(mock_case, BULK_FIELDS)
    y = mock_case.coordinates["y"]
    for side in ("cw", "hw"):
        fr = friction_reynolds(profiles, y, side)
        assert fr["u_tau"] > 0
        assert fr["Re_tau"] > 0


def test_case_physics_summary_has_expected_keys(mock_case):
    summary = case_physics_summary(mock_case)
    expected_keys = {
        "case_id", "rho_b", "U_b", "T_b", "P_b", "mu_b", "kappa_b", "cp_b",
        "c_b", "Re_b", "Pr_b", "Ec_b", "Br_b", "Ma_b",
        "T_cw", "u_tau_cw", "Re_tau_cw", "T_hw", "u_tau_hw", "Re_tau_hw",
    }
    assert expected_keys <= set(summary.keys())
    assert summary["case_id"] == "case_mock"

from pathlib import Path

import pytest

from metis.data.ingestion.hdf5_reader import HDF5Reader
from metis.testing.mock_dns import generate


@pytest.fixture
def mock_case(tmp_path) -> tuple[Path, Path]:
    return generate(tmp_path / "case_mock.h5", nx=4, ny=6, nz=4, seed=1)


def test_reader_reads_metadata(mock_case):
    h5_path, metadata_path = mock_case
    case = HDF5Reader().read(h5_path, metadata_path=metadata_path)
    assert case.metadata.case_id == "case_mock"
    assert case.metadata.Pb_Pc == pytest.approx(1.5)
    assert case.metadata.Thw_Tc == pytest.approx(1.1)
    assert case.metadata.grid == {"Nx": 4, "Ny": 6, "Nz": 4}


def test_reader_reads_fields_with_consistent_shapes(mock_case):
    h5_path, metadata_path = mock_case
    case = HDF5Reader().read(h5_path, metadata_path=metadata_path)
    assert {"u", "v", "w", "T", "rho", "avg_u", "rmsf_u", "favre_uffuff"} <= set(
        case.variable_names()
    )
    shapes = {v.shape for v in case.fields.values()}
    assert len(shapes) == 1
    assert case.fields["u"].shape == (6, 8, 6)  # (nx+2, ny+2, nz+2)
    assert case.coordinates["y"].shape == (8,)


def test_reader_raises_on_missing_required_variable(mock_case):
    h5_path, metadata_path = mock_case
    reader = HDF5Reader(required_variables=["u", "does_not_exist"])
    with pytest.raises(ValueError, match="missing variables"):
        reader.read(h5_path, metadata_path=metadata_path)


def test_reader_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        HDF5Reader().read(tmp_path / "does_not_exist.h5")


def test_reader_raises_on_missing_metadata(tmp_path):
    h5_path, metadata_path = generate(tmp_path / "case_mock.h5", nx=4, ny=4, nz=4, seed=1)
    metadata_path.unlink()
    with pytest.raises(FileNotFoundError):
        HDF5Reader().read(h5_path)


def test_reader_is_deterministic_for_a_fixed_seed(tmp_path):
    a_h5, a_meta = generate(tmp_path / "a.h5", nx=4, ny=4, nz=4, seed=7)
    b_h5, b_meta = generate(tmp_path / "b.h5", nx=4, ny=4, nz=4, seed=7)
    case_a = HDF5Reader().read(a_h5, metadata_path=a_meta)
    case_b = HDF5Reader().read(b_h5, metadata_path=b_meta)
    assert (case_a.fields["u"] == case_b.fields["u"]).all()

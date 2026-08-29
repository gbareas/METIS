import json

import pytest

from metis.data.ingestion.slice_reader import SliceReader
from metis.testing.mock_slices import generate


@pytest.fixture
def mock_slice(tmp_path):
    return generate(tmp_path / "case01" / "s3_center", nx=8, nz=8, n_snapshots=10, seed=1)


def test_reader_reads_metadata(mock_slice):
    case = SliceReader().read(mock_slice)
    assert case.metadata.case_id == "case_mock"
    assert case.metadata.slice_id == "s_mock"
    assert case.metadata.Pb_Pc == pytest.approx(1.5)
    assert case.metadata.grid == {"NX": 8, "NZ": 8}


def test_reader_reads_snapshots_and_mean(mock_slice):
    case = SliceReader().read(mock_slice)
    assert set(case.variable_names()) == {"u", "T", "cp"}
    assert case.snapshots["u"].shape == (10, 8, 8)
    assert case.mean["u"].shape == (8, 8)
    assert case.coordinates["x"].shape == (8,)
    assert case.coordinates["z"].shape == (8,)


def test_reader_rejects_non_xz_slice_prefix(mock_slice):
    meta_path = mock_slice / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["slice_prefix"] = "plane_XY_slice_1"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="only supports XZ"):
        SliceReader().read(mock_slice)


def test_reader_raises_on_missing_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        SliceReader().read(tmp_path / "does_not_exist")

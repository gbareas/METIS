"""Tests for metis.data.registry — CaseDescriptor / CaseRegistry (R2)."""
import json
import shutil

import pytest

from metis.data.ingestion.hdf5_reader import DNSCase
from metis.data.ingestion.slice_reader import SliceCase
from metis.data.registry import CaseDescriptor, CaseRegistry
from metis.features.regime import case_grid_labels
from metis.testing import mock_dns, mock_slices


def _make_case(data_root, case_id, *, seed, Pb_Pc, Thw_Tc, Tcw_Tc, slices=()):
    """Lay one case out under raw/ + processed/ (+ processed_slices/)."""
    raw_dir = data_root / "raw" / case_id
    _h5, meta = mock_dns.generate(
        raw_dir / f"snap_{seed:07d}.h5",
        nx=6, ny=8, nz=6, seed=seed,
        case_id=case_id, Pb_Pc=Pb_Pc, Thw_Tc=Thw_Tc, Tcw_Tc=Tcw_Tc,
    )
    processed_dir = data_root / "processed" / case_id
    processed_dir.mkdir(parents=True)
    shutil.copy(meta, processed_dir / "metadata.json")
    meta.unlink()
    for slice_id in slices:
        mock_slices.generate(
            data_root / "processed_slices" / case_id / slice_id,
            nx=8, nz=8, n_snapshots=12, seed=seed,
            case_id=case_id, slice_id=slice_id,
            Pb_Pc=Pb_Pc, Thw_Tc=Thw_Tc, Tcw_Tc=Tcw_Tc,
        )


@pytest.fixture
def data_root(tmp_path):
    _make_case(tmp_path, "case01", seed=1, Pb_Pc=1.5, Thw_Tc=1.1, Tcw_Tc=0.95,
               slices=("s3_center", "s2_max_u"))
    _make_case(tmp_path, "case02", seed=2, Pb_Pc=2.0, Thw_Tc=1.2, Tcw_Tc=0.90)  # no slices
    return tmp_path


@pytest.fixture
def registry(data_root):
    return CaseRegistry(data_root)


def test_discovers_cases_with_metadata(registry):
    assert registry.case_ids() == ["case01", "case02"]
    assert len(registry) == 2
    assert "case01" in registry
    assert "case99" not in registry


def test_descriptor_carries_ratios_and_grid(registry):
    d = registry["case01"]
    assert isinstance(d, CaseDescriptor)
    assert (d.Pb_Pc, d.Thw_Tc, d.Tcw_Tc) == (1.5, 1.1, 0.95)
    assert (d.nx, d.ny, d.nz) == (6, 8, 6)
    assert d.grid == {"Nx": 6, "Ny": 8, "Nz": 6}
    assert d.metadata_path == registry.processed_root / "case01" / "metadata.json"


def test_data_product_availability_flags(registry):
    c1, c2 = registry["case01"], registry["case02"]
    assert c1.has_raw and c1.has_processed and c1.has_slices
    assert c1.available_slices() == ["s2_max_u", "s3_center"]
    assert c2.has_raw and c2.has_processed
    assert not c2.has_slices
    assert c2.available_slices() == []
    assert c2.slice_root is None


def test_unknown_case_raises_keyerror_listing_known(registry):
    with pytest.raises(KeyError, match="case01"):
        registry["nope"]


def test_missing_metadata_raises(tmp_path):
    (tmp_path / "processed" / "case07").mkdir(parents=True)  # dir but no metadata.json
    reg = CaseRegistry(tmp_path)
    assert reg.case_ids() == []
    with pytest.raises(KeyError):
        reg["case07"]


def test_metadata_missing_required_key_raises(data_root):
    meta_path = data_root / "processed" / "case02" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["Tcw_Tc"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Tcw_Tc"):
        CaseRegistry(data_root)["case02"]


def test_require_reports_missing_slice_with_available_list(registry):
    with pytest.raises(FileNotFoundError, match="s2_max_u"):
        registry.require("case01", slices=["s3_center", "not_a_slice"])
    # a present one is fine and returns the descriptor
    assert registry.require("case01", raw=True, slices=["s3_center"]).case_id == "case01"


def test_require_reports_missing_raw(registry, data_root):
    for h5 in (data_root / "raw" / "case01").glob("*.h5"):
        h5.unlink()
    with pytest.raises(FileNotFoundError, match="raw"):
        registry.require("case01", raw=True)


def test_descriptor_loads_through_standard_readers(registry):
    d = registry["case01"]
    assert isinstance(d.load(), DNSCase)
    assert isinstance(d.load_slice("s3_center"), SliceCase)
    with pytest.raises(FileNotFoundError, match="no slice data"):
        registry["case02"].load_slice("s3_center")


def test_case_grid_labels_accepts_registry_and_matches_data_root(registry, data_root):
    ids = ("case01", "case02")
    from_reg = case_grid_labels(registry, ids)
    from_root = case_grid_labels(data_root, ids)
    assert from_reg == from_root
    assert from_reg["case01"].Pb_Pc == 1.5


def test_from_config_resolves_data_root(data_root, monkeypatch):
    monkeypatch.setenv("METIS_DATA_ROOT", str(data_root))
    reg = CaseRegistry.from_config(config={})
    assert reg.data_root == data_root
    assert reg.case_ids() == ["case01", "case02"]

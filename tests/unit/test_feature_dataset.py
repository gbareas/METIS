"""Tests for metis.data.datasets (R5): Level-1 FeatureDataset + caching."""
import shutil

import numpy as np
import pytest

from metis.data.datasets import FeatureDataset, build_feature_dataset
from metis.data.datasets import build as build_mod
from metis.data.registry import CaseRegistry
from metis.features.regime import FEATURE_NAMES, RICH_POD_SLICE_IDS
from metis.testing import mock_dns, mock_slices

CASE_IDS = ["case01", "case02", "case03"]


def _make_case(root, case_id, seed):
    raw = root / "raw" / case_id
    _h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=6, ny=8, nz=6, seed=seed,
                                  case_id=case_id, Pb_Pc=1.5 + 0.5 * seed, Thw_Tc=1.1,
                                  Tcw_Tc=0.95)
    proc = root / "processed" / case_id
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    for slice_id in RICH_POD_SLICE_IDS:
        mock_slices.generate(root / "processed_slices" / case_id / slice_id,
                             nx=8, nz=8, n_snapshots=12, seed=seed,
                             case_id=case_id, slice_id=slice_id)


@pytest.fixture
def registry(tmp_path):
    for i, cid in enumerate(CASE_IDS, start=1):
        _make_case(tmp_path, cid, seed=i)
    return CaseRegistry(tmp_path)


def _spy_on_builder(monkeypatch):
    calls = []
    real = build_mod.build_feature_matrix
    monkeypatch.setattr(
        build_mod, "build_feature_matrix",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    return calls


def test_build_compact_dataset_shape_and_names(registry):
    ds = build_feature_dataset(registry, CASE_IDS, "compact")
    assert isinstance(ds, FeatureDataset)
    assert ds.X.shape == (3, len(FEATURE_NAMES))
    assert ds.feature_names == list(FEATURE_NAMES)
    assert ds.case_ids == CASE_IDS
    assert ds.provenance["n_cases"] == 3
    assert ds.provenance["fingerprint"] and ds.provenance["code_version"]


def test_save_load_round_trip(registry, tmp_path):
    out = tmp_path / "art"
    ds = build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    reloaded = FeatureDataset.load(out)
    np.testing.assert_array_equal(reloaded.X, ds.X)
    assert reloaded.provenance == ds.provenance
    assert reloaded.feature_names == ds.feature_names


def test_second_build_is_a_cache_hit(registry, tmp_path, monkeypatch):
    calls = _spy_on_builder(monkeypatch)
    out = tmp_path / "art"

    ds1 = build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    assert len(calls) == 1

    ds2 = build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    assert len(calls) == 1  # not recomputed
    np.testing.assert_array_equal(ds1.X, ds2.X)
    assert ds1.provenance["created_at"] == ds2.provenance["created_at"]


def test_rebuild_flag_forces_recompute(registry, tmp_path, monkeypatch):
    calls = _spy_on_builder(monkeypatch)
    out = tmp_path / "art"
    build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out, rebuild=True)
    assert len(calls) == 2


def test_feature_code_change_invalidates_cache(registry, tmp_path, monkeypatch):
    calls = _spy_on_builder(monkeypatch)
    out = tmp_path / "art"

    monkeypatch.setattr(build_mod, "_feature_code_hash", lambda: "codehash-A")
    build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    assert len(calls) == 1

    monkeypatch.setattr(build_mod, "_feature_code_hash", lambda: "codehash-B")
    ds = build_feature_dataset(registry, CASE_IDS, "compact", out_dir=out)
    assert len(calls) == 2  # fingerprint changed -> recomputed
    assert ds.fingerprint == FeatureDataset.load(out).fingerprint


def test_fingerprint_depends_on_cases_and_feature_set(registry):
    a = build_feature_dataset(registry, CASE_IDS, "compact").fingerprint
    b = build_feature_dataset(registry, CASE_IDS[:2], "compact").fingerprint
    c = build_feature_dataset(registry, CASE_IDS, "bulk").fingerprint
    assert a != b and a != c and b != c


def test_row_and_matrix_access(registry):
    ds = build_feature_dataset(registry, CASE_IDS, "compact")
    np.testing.assert_array_equal(ds.row("case02"), ds.X[1])
    np.testing.assert_array_equal(ds.matrix(["case03", "case01"]), ds.X[[2, 0]])
    with pytest.raises(KeyError, match="case99"):
        ds.row("case99")


def test_constructor_validates_shapes():
    with pytest.raises(ValueError, match="rows"):
        FeatureDataset(X=np.zeros((2, 3)), case_ids=["a"], feature_names=["f0", "f1", "f2"],
                       feature_set="compact")

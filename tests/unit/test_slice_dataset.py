"""Tests for metis.data.datasets.slice_dataset (I2-B / B2)."""
import shutil

import numpy as np
import pytest

from metis.data.datasets import (
    SliceDataset,
    build_slice_dataset,
    normalize_split_config,
)
from metis.data.registry import CaseRegistry
from metis.testing import mock_dns, mock_slices

SLICE_ID = "s3_center"
N_SNAP = 20


def _make_case(root, case_id, seed):
    raw = root / "raw" / case_id
    _h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=6, ny=8, nz=6, seed=seed,
                                  case_id=case_id)
    proc = root / "processed" / case_id
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    mock_slices.generate(root / "processed_slices" / case_id / SLICE_ID, nx=8, nz=8,
                         n_snapshots=N_SNAP, seed=seed, case_id=case_id, slice_id=SLICE_ID)


@pytest.fixture
def registry(tmp_path):
    for i, cid in enumerate(("case01", "case02", "case10", "case15"), start=1):
        _make_case(tmp_path, cid, seed=i)
    return CaseRegistry(tmp_path)


@pytest.fixture
def splits():
    return normalize_split_config({
        "train_cases": ["case01", "case02"],
        "train_snapshots": [0, 15],
        "val_snapshots": [15, 20],
        "ood_cases": ["case10", "case15"],
        "ood_snapshots": [0, 20],
    })


def test_split_shapes_ids_and_indices(registry, splits, tmp_path):
    ds = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=tmp_path / "sd")
    assert ds["train"].X.shape == (2 * 15, 8, 8)
    assert ds["val"].X.shape == (2 * 5, 8, 8)
    assert ds["ood"].X.shape == (2 * 20, 8, 8)

    tr = ds["train"]
    assert set(tr.case_ids) == {"case01", "case02"}
    assert tr.snapshot_idx.min() == 0 and tr.snapshot_idx.max() == 14
    assert ds["val"].snapshot_idx.min() == 15
    assert set(ds["ood"].case_ids) == {"case10", "case15"}


def test_scaler_is_fit_on_train_only_and_shared(registry, splits, tmp_path):
    ds = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=tmp_path / "sd")
    assert ds["train"].scaler["mean"] == pytest.approx(float(ds["train"].X.mean()))
    assert ds["train"].scaler["std"] == pytest.approx(float(ds["train"].X.std()))
    assert ds["val"].scaler == ds["train"].scaler == ds["ood"].scaler
    # not the combined mean
    combined = np.concatenate([ds["train"].X.ravel(), ds["ood"].X.ravel()]).mean()
    assert not np.isclose(ds["train"].scaler["mean"], combined)


def test_standardize_inverse_round_trip(registry, splits, tmp_path):
    ds = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=tmp_path / "sd")["val"]
    np.testing.assert_allclose(ds.inverse(ds.standardize()), ds.X, atol=1e-4)
    assert ds.standardize().std() == pytest.approx(
        ds.X.std() / ds.scaler["std"], rel=1e-5)


def test_save_load_round_trip(registry, splits, tmp_path):
    out = tmp_path / "sd"
    built = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=out)
    reloaded = SliceDataset.load(out, "ood")
    np.testing.assert_array_equal(reloaded.X, built["ood"].X)
    assert reloaded.field == "u" and reloaded.slice_id == SLICE_ID
    assert reloaded.provenance["n"]["ood"] == len(reloaded)


def test_second_build_is_a_cache_hit(registry, splits, tmp_path):
    out = tmp_path / "sd"
    a = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=out)
    b = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=out)
    assert a["train"].provenance["created_at"] == b["train"].provenance["created_at"]


def test_touching_a_slice_file_invalidates_the_cache(registry, splits, tmp_path):
    import os

    out = tmp_path / "sd"
    a = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=out)
    f = registry["case01"].slice_root / SLICE_ID / "snapshots_u.npy"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    b = build_slice_dataset(registry, "u", SLICE_ID, splits, out_dir=out)
    assert a["train"].provenance["fingerprint"] != b["train"].provenance["fingerprint"]


def test_normalize_split_config_shape():
    s = normalize_split_config({
        "train_cases": ["c1"], "train_snapshots": [0, 4],
        "val_snapshots": [4, 5], "ood_cases": ["c9"], "ood_snapshots": [0, 5],
    })
    assert s["train"]["cases"] == ["c1"] and s["train"]["snapshots"] == (0, 4)
    assert s["val"]["cases"] == ["c1"] and s["val"]["snapshots"] == (4, 5)
    assert s["ood"]["cases"] == ["c9"]

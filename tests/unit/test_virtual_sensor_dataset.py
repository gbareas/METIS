"""Tests for metis.data.datasets.virtual_sensor_dataset (Phase C / C2)."""
import shutil

import numpy as np
import pytest

from metis.data.datasets import (
    VirtualSensorDataset,
    build_virtual_sensor_dataset,
    compute_sensor_channels,
    normalize_vs_split_config,
)
from metis.data.registry import CaseRegistry
from metis.testing import mock_dns, mock_slices

N_SNAP = 60
PROBES = {"pa": {"slice_id": "sa"}, "pb": {"slice_id": "sb"}}
CONFIG = {
    "name": "vs_test",
    "probes": PROBES,
    "channels": {"fields": ["u", "T", "cp"],
                 "aggregates": {"pmf": "plane_mean_fluctuation", "rms": "plane_rms_fluctuation"}},
    "task": {"context_length": 8, "horizons": [1, 4]},
}


def _make_case(root, cid, seed):
    raw = root / "raw" / cid
    _h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=6, ny=8, nz=6, seed=seed, case_id=cid)
    proc = root / "processed" / cid
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    for j, (probe, spec) in enumerate(PROBES.items()):
        mock_slices.generate(
            root / "processed_slices" / cid / spec["slice_id"],
            nx=6, nz=6, n_snapshots=N_SNAP, seed=10 * seed + j,
            case_id=cid, slice_id=spec["slice_id"], y_loc=1e-4 * (1 + j),
        )


@pytest.fixture
def registry(tmp_path):
    for i, cid in enumerate(("case01", "case02", "case10"), start=1):
        _make_case(tmp_path, cid, seed=i)
    return CaseRegistry(tmp_path)


@pytest.fixture
def splits():
    return normalize_vs_split_config({
        "train_cases": ["case01", "case02"], "train_steps": [0, 40],
        "val_steps": [40, 60], "ood_cases": ["case10"], "ood_steps": [0, 60],
    })


# --- compute_sensor_channels --------------------------------------
def test_compute_sensor_channels_values_and_names():
    rng = np.random.default_rng(0)
    fl = {"u": rng.normal(size=(5, 4, 4)), "T": rng.normal(size=(5, 4, 4)),
          "cp": rng.normal(size=(5, 4, 4))}
    mat, names = compute_sensor_channels(fl)
    assert mat.shape == (5, 6)
    assert names == ["u_pmf", "u_rms", "T_pmf", "T_rms", "cp_pmf", "cp_rms"]
    flat = fl["u"].reshape(5, -1)
    np.testing.assert_allclose(mat[:, 0], flat.mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(mat[:, 1], np.sqrt((flat**2).mean(axis=1)), rtol=1e-5)


# --- build --------------------------------------------------------
def test_build_shapes_channels_and_static_context(registry, splits, tmp_path):
    ds = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=tmp_path / "vs")
    tr = ds["train"]
    assert tr.n_channels == 2 * 3 * 2 == len(tr.channels)
    assert tr.channels[:3] == ["pa.u_pmf", "pa.u_rms", "pa.T_pmf"]
    assert tr.X.shape == (2 * 40, 12)
    assert ds["val"].X.shape == (2 * 20, 12)
    assert ds["ood"].X.shape == (1 * 60, 12)
    assert set(tr.static_context) == {"case01", "case02", "case10"}
    assert set(tr.static_context["case01"]["y_loc"]) == {"pa", "pb"}


def test_scaler_is_train_only_and_shared(registry, splits, tmp_path):
    ds = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=tmp_path / "vs")
    tr = ds["train"]
    np.testing.assert_allclose(tr.scaler["mean"], tr.X.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(tr.scaler["std"], tr.X.std(axis=0), rtol=1e-4)
    assert ds["val"].scaler == tr.scaler == ds["ood"].scaler
    z = tr.standardize()
    assert np.abs(z.mean(axis=0)).max() < 1e-4
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-4)


# --- windowing / leakage ---------------------------------------
def test_window_counts_and_assemble_shapes(registry, splits, tmp_path):
    ds = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=tmp_path / "vs")
    lgth, hmax = 8, 4
    assert len(ds["train"]) == 2 * (40 - lgth - hmax + 1)
    assert len(ds["val"]) == 2 * (20 - lgth - hmax + 1)
    assert len(ds["ood"]) == 1 * (60 - lgth - hmax + 1)
    Xc, Y, _ac, anchor_step = ds["train"].assemble()
    assert Xc.shape == (len(ds["train"]), lgth, 12)
    assert Y.shape == (len(ds["train"]), 2, 12)
    assert anchor_step.min() == lgth - 1


def test_no_window_crosses_a_case_or_split_boundary(registry, splits, tmp_path):
    ds = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=tmp_path / "vs")
    for split in ("train", "val", "ood"):
        d = ds[split]
        for t in d.window_anchors:
            rows = list(range(t - d.context_length + 1, t + 1)) + [t + h for h in d.horizons]
            assert len(set(d.case_ids[rows])) == 1                # one case
    assert ds["train"].step_idx.max() < 40                        # train steps only
    assert ds["val"].step_idx.min() == 40                         # val steps only
    # a val context window never reaches back into train steps
    d = ds["val"]
    earliest = d.step_idx[d.window_anchors.min() - d.context_length + 1]
    assert earliest >= 40


# --- cache -----------------------------------------------------
def test_cache_hit_and_bust_on_source_change(registry, splits, tmp_path):
    out = tmp_path / "vs"
    fp1 = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=out)["train"].provenance
    mtime1 = (out / "train.npz").stat().st_mtime_ns

    fp2 = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=out)["train"].provenance
    assert fp2["fingerprint"] == fp1["fingerprint"]
    assert (out / "train.npz").stat().st_mtime_ns == mtime1       # cache hit: not rewritten

    snap = registry["case01"].slice_root / "sa" / "snapshots_u.npy"
    snap.write_bytes(snap.read_bytes())                            # bump mtime + size-stable content
    fp3 = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=out)["train"].provenance
    assert fp3["fingerprint"] != fp1["fingerprint"]               # manifest changed -> rebuilt
    assert (out / "train.npz").stat().st_mtime_ns != mtime1


def test_load_round_trip(registry, splits, tmp_path):
    out = tmp_path / "vs"
    built = build_virtual_sensor_dataset(registry, CONFIG, splits, out_dir=out)
    reloaded = VirtualSensorDataset.load(out, "ood")
    np.testing.assert_array_equal(reloaded.X, built["ood"].X)
    np.testing.assert_array_equal(reloaded.window_anchors, built["ood"].window_anchors)
    assert reloaded.channels == built["ood"].channels
    assert reloaded.horizons == [1, 4]

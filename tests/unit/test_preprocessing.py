"""Tests for metis.data.preprocessing (R4)."""
import numpy as np
import pytest

from metis.data.preprocessing import (
    MinMaxScaler,
    NotFittedError,
    StandardScaler,
    interior_fields,
    load,
    save,
    select_variables,
    strip_ghost_cells,
    subsample,
)
from metis.evaluation.regime import standardize
from metis.testing import mock_dns


# --- StandardScaler ------------------------------------------------
def test_standard_scaler_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(50, 4))
    Xs = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0), 1, atol=1e-12)


def test_standard_scaler_inverse_round_trips():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3)) * [10, 0.1, 100]
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-9)


def test_standard_scaler_matches_regime_standardize():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(11, 7))
    np.testing.assert_allclose(StandardScaler().fit_transform(X), standardize(X))


def test_standard_scaler_leaves_zero_variance_column_at_zero():
    X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    Xs = StandardScaler().fit_transform(X)
    assert np.all(Xs[:, 0] == 0.0)


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(np.zeros((2, 2)))


# --- leakage control --------------------------------------------
def test_fit_on_train_only_no_leakage_from_test():
    train = np.array([[0.0], [2.0], [4.0]])       # train mean 2, std sqrt(8/3)
    test = np.array([[100.0], [200.0]])           # wildly different distribution
    sc = StandardScaler().fit(train)
    fitted_mean = sc.mean_.copy()

    _ = sc.transform(test)                        # applying to test must not refit
    np.testing.assert_array_equal(sc.mean_, fitted_mean)
    np.testing.assert_allclose(sc.mean_, train.mean(axis=0))
    assert not np.allclose(sc.mean_, np.concatenate([train, test]).mean(axis=0))
    # a train sample at the train mean maps to exactly 0
    np.testing.assert_allclose(sc.transform([[2.0]]), [[0.0]])


# --- serialization ----------------------------------------------
def test_standard_scaler_params_round_trip():
    from metis.data.preprocessing import from_dict, to_dict

    sc = StandardScaler().fit(np.random.default_rng(3).normal(size=(8, 5)))
    clone = from_dict(to_dict(sc))
    assert isinstance(clone, StandardScaler)
    np.testing.assert_allclose(clone.mean_, sc.mean_)
    np.testing.assert_allclose(clone.std_, sc.std_)


def test_save_and_load_fitted_scaler(tmp_path):
    X = np.random.default_rng(4).normal(size=(10, 3))
    sc = StandardScaler().fit(X)
    path = save(sc, tmp_path / "scaler.json")
    reloaded = load(path)
    np.testing.assert_allclose(reloaded.transform(X), sc.transform(X))


# --- MinMaxScaler ---------------------------------------------
def test_minmax_scaler_range_and_inverse():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 4)) * 20 + 3
    sc = MinMaxScaler().fit(X)
    Xs = sc.transform(X)
    assert Xs.min() >= 0.0 and Xs.max() <= 1.0
    np.testing.assert_allclose(Xs.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xs.max(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(sc.inverse_transform(Xs), X, atol=1e-9)


def test_minmax_constant_column_is_safe():
    X = np.array([[2.0, 1.0], [2.0, 3.0]])
    Xs = MinMaxScaler().fit_transform(X)
    assert np.all(np.isfinite(Xs))
    assert np.all(Xs[:, 0] == 0.0)


# --- stateless field helpers -----------------------------------
def test_strip_ghost_cells_shape():
    arr = np.zeros((10, 12, 8))
    assert strip_ghost_cells(arr).shape == (8, 10, 6)
    assert strip_ghost_cells(arr, 0).shape == (10, 12, 8)
    assert strip_ghost_cells(arr, 2).shape == (6, 8, 4)


def test_select_variables_picks_and_orders():
    fields = {"u": np.zeros(2), "T": np.ones(2), "rho": np.full(2, 3.0)}
    out = select_variables(fields, ["rho", "u"])
    assert list(out) == ["rho", "u"]
    with pytest.raises(KeyError, match="nope"):
        select_variables(fields, ["u", "nope"])


def test_subsample_stride():
    arr = np.arange(24).reshape(4, 6)
    assert subsample(arr, 2).shape == (2, 3)
    assert subsample(arr, 2, axes=[1]).shape == (4, 3)
    assert subsample(arr, 1).shape == (4, 6)


def test_interior_fields_on_mock_case(tmp_path):
    from metis.data.ingestion.hdf5_reader import HDF5Reader

    h5, meta = mock_dns.generate(tmp_path / "c.h5", nx=4, ny=6, nz=5, seed=1)
    case = HDF5Reader().read(h5, metadata_path=meta)
    interior = interior_fields(case)
    assert interior["u"].shape == (5, 6, 4)  # (Nz, Ny, Nx), ghosts removed

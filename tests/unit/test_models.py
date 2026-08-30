"""Tests for metis.models — PCARepresentation (I2/I4)."""
import numpy as np
import pytest

from metis.data.preprocessing import from_dict, to_dict
from metis.evaluation.regime import pca as regime_pca
from metis.models import PCARepresentation, RepresentationModel


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 8)) @ np.diag([5, 4, 3, 1, 0.5, 0.2, 0.1, 0.05])


# --- PCA baseline ------------------------------------------------
def test_pca_transform_shape_and_is_a_representation_model(X):
    m = PCARepresentation(latent_dim=3).fit(X)
    assert isinstance(m, RepresentationModel)
    assert m.transform(X).shape == (20, 3)
    assert m.latent_dim == 3


def test_pca_scores_match_regime_pca(X):
    scores, _, _ = regime_pca(X, n_components=2)
    ours = PCARepresentation(latent_dim=2).fit(X).transform(X)
    # sign of a PC is arbitrary; compare up to per-column sign
    np.testing.assert_allclose(np.abs(ours), np.abs(scores), atol=1e-9)


def test_pca_full_rank_reconstruction_is_exact(X):
    m = PCARepresentation(latent_dim=8).fit(X)
    np.testing.assert_allclose(m.inverse_transform(m.transform(X)), X, atol=1e-8)


def test_pca_reconstruction_mse_decreases_with_latent_dim(X):
    e1 = PCARepresentation(latent_dim=1).fit(X).reconstruction_mse(X)
    e4 = PCARepresentation(latent_dim=4).fit(X).reconstruction_mse(X)
    assert e4 < e1


def test_pca_save_load_round_trip(X):
    m = PCARepresentation(latent_dim=2).fit(X)
    clone = from_dict(to_dict(m))
    assert isinstance(clone, PCARepresentation)
    np.testing.assert_allclose(clone.transform(X), m.transform(X))

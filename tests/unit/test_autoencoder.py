"""Tests for metis.models.autoencoder (I2/I4) — needs torch (ml extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metis.data.preprocessing import from_dict, to_dict
from metis.data.preprocessing.base import NotFittedError
from metis.models.autoencoder import Autoencoder
from metis.training import Trainer

_FAST = Trainer(max_epochs=300, patience=80, log_every=50, seed=0)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 8)) @ np.diag([5, 4, 3, 1, 0.5, 0.2, 0.1, 0.05])


def test_autoencoder_fits_and_transforms(X):
    ae = Autoencoder(latent_dim=2, hidden=(16,)).fit(X, trainer=_FAST)
    assert ae.is_fitted
    assert ae.transform(X).shape == (20, 2)
    assert ae.inverse_transform(ae.transform(X)).shape == X.shape
    assert ae.history_["best_train_mse"] < ae.history_["logged_losses"][0]


def test_autoencoder_is_deterministic_for_a_fixed_seed(X):
    z1 = Autoencoder(latent_dim=2, hidden=(16,)).fit(X, trainer=_FAST).transform(X)
    z2 = Autoencoder(latent_dim=2, hidden=(16,)).fit(X, trainer=_FAST).transform(X)
    np.testing.assert_allclose(z1, z2, atol=1e-6)


def _init_state(seed):
    from metis.models.autoencoder import _make_net

    net = _make_net(8, 2, (16,), "tanh", seed=seed)
    return {k: v.clone() for k, v in net.state_dict().items()}


def test_same_seed_gives_same_initial_parameters():
    a, b = _init_state(7), _init_state(7)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_different_seeds_give_different_initial_parameters():
    a, b = _init_state(0), _init_state(1)
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_different_trainer_seeds_give_different_latents(X):
    z0 = Autoencoder(latent_dim=2, hidden=(16,)).fit(
        X, trainer=Trainer(max_epochs=300, patience=80, log_every=50, seed=0)).transform(X)
    z1 = Autoencoder(latent_dim=2, hidden=(16,)).fit(
        X, trainer=Trainer(max_epochs=300, patience=80, log_every=50, seed=1)).transform(X)
    assert not np.allclose(z0, z1, atol=1e-4)


def test_autoencoder_rejects_bad_activation():
    with pytest.raises(ValueError, match="activation"):
        Autoencoder(activation="sigmoid")


def test_autoencoder_save_load_round_trip(X):
    ae = Autoencoder(latent_dim=2, hidden=(16,)).fit(X, trainer=_FAST)
    clone = from_dict(to_dict(ae))
    assert isinstance(clone, Autoencoder)
    np.testing.assert_allclose(clone.transform(X), ae.transform(X), atol=1e-5)
    np.testing.assert_allclose(
        clone.inverse_transform(clone.transform(X)),
        ae.inverse_transform(ae.transform(X)),
        atol=1e-5,
    )


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        Autoencoder().transform(np.zeros((2, 4)))

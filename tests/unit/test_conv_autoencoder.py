"""Tests for metis.models.conv_autoencoder (I2-B / B4) — needs torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from metis.data.preprocessing import from_dict, to_dict
from metis.data.preprocessing.base import NotFittedError
from metis.models.conv_autoencoder import ConvAutoencoder, _make_conv_net
from metis.training import Trainer

_FAST = Trainer(max_epochs=40, patience=40, log_every=10, seed=0, batch_size=8)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    # low-rank structured field so a tiny latent can actually reconstruct it
    loadings = rng.normal(size=(24, 3))
    modes = rng.normal(size=(3, 16 * 16))
    field = (loadings @ modes).reshape(24, 16, 16)
    return (field + 0.05 * rng.normal(size=(24, 16, 16))).astype(np.float32)


def _model(latent_dim=4):
    return ConvAutoencoder(latent_dim=latent_dim, input_hw=(16, 16), channels=(4, 8, 16))


def test_fits_and_transforms(X):
    ae = _model().fit(X, trainer=_FAST)
    assert ae.is_fitted
    assert ae.transform(X).shape == (24, 4)
    assert ae.inverse_transform(ae.transform(X)).shape == X.shape
    assert ae.history_["logged_losses"][0] > ae.history_["best_train_mse"]


def test_deterministic_for_fixed_seed(X):
    z1 = _model().fit(X, trainer=_FAST).transform(X)
    z2 = _model().fit(X, trainer=_FAST).transform(X)
    np.testing.assert_allclose(z1, z2, atol=1e-6)


def test_different_seeds_give_different_latents(X):
    z0 = _model().fit(X, trainer=Trainer(max_epochs=40, log_every=10, seed=0, batch_size=8)).transform(X)
    z1 = _model().fit(X, trainer=Trainer(max_epochs=40, log_every=10, seed=1, batch_size=8)).transform(X)
    assert not np.allclose(z0, z1, atol=1e-4)


def test_seed_drives_initial_weights():
    a = _make_conv_net((16, 16), 4, (4, 8, 16), "gelu", seed=5)
    b = _make_conv_net((16, 16), 4, (4, 8, 16), "gelu", seed=5)
    c = _make_conv_net((16, 16), 4, (4, 8, 16), "gelu", seed=6)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_rejects_bad_activation():
    with pytest.raises(ValueError, match="activation"):
        ConvAutoencoder(activation="sigmoid")


def test_rejects_input_hw_not_divisible_by_eight():
    ae = ConvAutoencoder(latent_dim=4, input_hw=(20, 16), channels=(4, 8, 16))
    with pytest.raises(ValueError, match="divisible by 8"):
        ae.fit(np.zeros((4, 20, 16), dtype=np.float32), trainer=_FAST)


def test_rejects_input_shape_mismatch(X):
    with pytest.raises(ValueError, match="does not match input_hw"):
        _model().fit(np.zeros((4, 8, 8), dtype=np.float32), trainer=_FAST)


def test_rejects_non_3d_input(X):
    with pytest.raises(ValueError, match=r"\(n, H, W\)"):
        _model().fit(X.reshape(24, -1), trainer=_FAST)


def test_validation_split_early_stops(X):
    ae = _model().fit(
        X[:18], X_val=X[18:], trainer=Trainer(max_epochs=100000, patience=8, min_delta=1e-3,
                                              log_every=200, seed=0, batch_size=8)
    )
    assert ae.history_["monitor"] == "val_mse"
    assert ae.history_["epochs_run"] < 100000


def test_standardize_round_trips(X):
    ae = ConvAutoencoder(latent_dim=4, input_hw=(16, 16), channels=(4, 8, 16),
                         standardize=True).fit(X, trainer=_FAST)
    assert ae.scaler_ is not None
    assert ae.inverse_transform(ae.transform(X)).shape == X.shape


def test_save_load_round_trip(X):
    ae = _model().fit(X, trainer=_FAST)
    clone = from_dict(to_dict(ae))
    assert isinstance(clone, ConvAutoencoder)
    np.testing.assert_allclose(clone.transform(X), ae.transform(X), atol=1e-5)
    np.testing.assert_allclose(
        clone.inverse_transform(clone.transform(X)),
        ae.inverse_transform(ae.transform(X)),
        atol=1e-5,
    )


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        _model().transform(np.zeros((2, 16, 16), dtype=np.float32))

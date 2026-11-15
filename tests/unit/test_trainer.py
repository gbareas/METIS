"""Tests for metis.training.Trainer (I3)."""
import numpy as np
import pytest

pytest.importorskip("torch")

from metis.models.autoencoder import Autoencoder
from metis.training import Trainer


@pytest.fixture
def X():
    rng = np.random.default_rng(1)
    return rng.normal(size=(24, 6))


def test_fit_reduces_loss_and_reports_history(X):
    ae = Autoencoder(latent_dim=2, hidden=(12,))
    hist = Trainer(max_epochs=400, patience=100, log_every=50, seed=0).fit(ae, ae_input(ae, X))
    assert hist["best_train_mse"] < hist["logged_losses"][0]
    assert hist["best_epoch"] <= hist["epochs_run"]


def ae_input(ae, X):
    # Trainer works on already-scaled input; mimic Autoencoder.fit's wiring
    from metis.data.preprocessing import StandardScaler

    ae.n_features_ = X.shape[1]
    ae.scaler_ = StandardScaler().fit(X)
    from metis.models.autoencoder import _make_net

    ae.module = _make_net(X.shape[1], ae.latent_dim, ae.hidden, ae.activation, seed=0)
    return ae.scaler_.transform(X)


def test_early_stopping_triggers_before_max_epochs(X):
    ae = Autoencoder(latent_dim=3, hidden=(12,))
    hist = Trainer(max_epochs=100000, patience=30, min_delta=1e-4, log_every=100, seed=0).fit(
        ae, ae_input(ae, X)
    )
    assert hist["epochs_run"] < 100000


def test_checkpoint_is_written(tmp_path, X):
    import torch

    ae = Autoencoder(latent_dim=2, hidden=(8,))
    hist = Trainer(max_epochs=200, patience=200, log_every=100, seed=0,
                   checkpoint_dir=tmp_path / "ckpt").fit(ae, ae_input(ae, X))
    ckpt = torch.load(hist["checkpoint"], weights_only=False)
    assert "model_state" in ckpt and ckpt["epoch"] == hist["best_epoch"]


def test_deterministic_for_fixed_seed(X):
    def run():
        ae = Autoencoder(latent_dim=2, hidden=(8,))
        h = Trainer(max_epochs=200, patience=200, log_every=100, seed=7).fit(ae, ae_input(ae, X))
        return h["best_train_mse"]

    assert run() == pytest.approx(run(), abs=1e-9)


def test_minibatch_reduces_loss_and_is_deterministic(X):
    def run():
        ae = Autoencoder(latent_dim=2, hidden=(12,))
        return Trainer(max_epochs=300, patience=300, log_every=50, seed=3,
                       batch_size=8).fit(ae, ae_input(ae, X))

    h = run()
    assert h["best_train_mse"] < h["logged_losses"][0]
    assert run()["best_train_mse"] == pytest.approx(h["best_train_mse"], abs=1e-9)


def test_validation_monitor_switches_history_keys(X):
    ae = Autoencoder(latent_dim=2, hidden=(12,))
    Xs = ae_input(ae, X)
    hist = Trainer(max_epochs=300, patience=300, log_every=50, seed=0, batch_size=8).fit(
        ae, Xs[:16], X_val=Xs[16:]
    )
    assert hist["monitor"] == "val_mse"
    assert "best_val_mse" in hist
    assert "best_train_mse" in hist  # stable key kept for back-compat
    assert hist["logged_losses"][0] > hist["best_train_mse"]


def test_early_stopping_uses_validation_loss(X):
    ae = Autoencoder(latent_dim=3, hidden=(12,))
    Xs = ae_input(ae, X)
    hist = Trainer(max_epochs=100000, patience=20, min_delta=1e-3, log_every=200, seed=0).fit(
        ae, Xs[:16], X_val=Xs[16:]
    )
    assert hist["epochs_run"] < 100000
    assert hist["best_epoch"] <= hist["epochs_run"]

"""Small MLP autoencoder for representation learning (milestone I2 / I4).

Needs torch (the `ml` extra). Encoder/decoder are symmetric MLPs; the
input is standardised on `fit` and de-standardised on reconstruction. The
optimisation loop lives in `metis.training.Trainer`, so `fit` just wires
the two together — `model.fit(X)` works standalone, or pass an explicit
`Trainer` / a `metis.tracking` run for logged training.
"""
from __future__ import annotations

import numpy as np

from metis.data.preprocessing import StandardScaler
from metis.models.base import RepresentationModel

_ACTIVATIONS = ("tanh", "relu", "gelu")


def _make_net(n_features: int, latent_dim: int, hidden: tuple[int, ...], activation: str):
    import torch
    from torch import nn

    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

    def mlp(sizes: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        return nn.Sequential(*layers)

    enc_sizes = [n_features, *hidden, latent_dim]
    dec_sizes = [latent_dim, *reversed(hidden), n_features]

    class _AENet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = mlp(enc_sizes)
            self.decoder = mlp(dec_sizes)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    torch.manual_seed(0)  # deterministic init; Trainer re-seeds before the loop
    return _AENet()


class Autoencoder(RepresentationModel):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden: tuple[int, ...] = (64,),
        *,
        activation: str = "tanh",
        standardize: bool = True,
        trainer=None,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}")
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = activation
        self.standardize = bool(standardize)
        self._trainer = trainer
        self.module = None
        self.scaler_: StandardScaler | None = None
        self.n_features_: int | None = None
        self.history_: dict | None = None

    @property
    def is_fitted(self) -> bool:
        return self.module is not None and self.history_ is not None

    def _default_trainer(self):
        if self._trainer is not None:
            return self._trainer
        from metis.training.trainer import Trainer

        return Trainer()

    def fit(self, X, *, run=None, trainer=None) -> Autoencoder:
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Xs = self.scaler_.transform(X) if self.scaler_ else X
        self.module = _make_net(self.n_features_, self.latent_dim, self.hidden, self.activation)
        self.history_ = (trainer or self._default_trainer()).fit(self, Xs, run=run)
        return self

    # -- inference -------------------------------------------------
    def _to_tensor(self, arr):
        import torch

        return torch.as_tensor(np.asarray(arr, dtype=np.float32))

    def transform(self, X):
        self._check_fitted()
        import torch

        Xs = self.scaler_.transform(X) if self.scaler_ else np.asarray(X, dtype=float)
        with torch.no_grad():
            return self.module.encode(self._to_tensor(Xs)).cpu().numpy()

    def inverse_transform(self, Z):
        self._check_fitted()
        import torch

        with torch.no_grad():
            Xs = self.module.decode(self._to_tensor(Z)).cpu().numpy()
        return self.scaler_.inverse_transform(Xs) if self.scaler_ else Xs

    # -- serialisation -------------------------------------------
    def get_params(self) -> dict:
        state = None
        if self.module is not None:
            state = {k: v.cpu().tolist() for k, v in self.module.state_dict().items()}
        return {
            "latent_dim": self.latent_dim,
            "hidden": list(self.hidden),
            "activation": self.activation,
            "standardize": self.standardize,
            "n_features": self.n_features_,
            "scaler": None if self.scaler_ is None else self.scaler_.get_params(),
            "state_dict": state,
            "history": self.history_,
        }

    @classmethod
    def from_params(cls, params: dict) -> Autoencoder:
        obj = cls(
            latent_dim=params["latent_dim"],
            hidden=tuple(params["hidden"]),
            activation=params["activation"],
            standardize=params["standardize"],
        )
        obj.n_features_ = params.get("n_features")
        if params.get("scaler") is not None:
            obj.scaler_ = StandardScaler.from_params(params["scaler"])
        if params.get("state_dict") is not None:
            import torch

            obj.module = _make_net(
                obj.n_features_, obj.latent_dim, obj.hidden, obj.activation
            )
            obj.module.load_state_dict(
                {k: torch.tensor(v) for k, v in params["state_dict"].items()}
            )
        obj.history_ = params.get("history")
        return obj

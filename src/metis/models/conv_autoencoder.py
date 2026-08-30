"""Small convolutional autoencoder for 2-D field snapshots (I2-B / B4).

Fixed architecture per `docs/i2b_representation_protocol.md` §4: three
stride-2 conv blocks (H → H/8), a linear bottleneck to `latent_dim`, and
a mirrored transpose-conv decoder. Needs torch (the `ml` extra). Operates
on `(n, H, W)` arrays; the optimisation loop lives in
`metis.training.Trainer`.
"""
from __future__ import annotations

import numpy as np

from metis.data.preprocessing import StandardScaler
from metis.models.base import RepresentationModel

_ACTIVATIONS = ("gelu", "relu", "tanh")


def _make_conv_net(input_hw, latent_dim, channels, activation, seed):
    import torch
    from torch import nn

    act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    h, w = input_hw
    if h % 8 or w % 8:
        raise ValueError(f"input {input_hw} must be divisible by 8 (three stride-2 blocks)")
    fh, fw = h // 8, w // 8
    last = channels[-1]
    flat = last * fh * fw

    class _ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            enc, cin = [], 1
            for c in channels:
                enc += [nn.Conv2d(cin, c, 3, stride=2, padding=1), act()]
                cin = c
            self.enc_conv = nn.Sequential(*enc)
            self.enc_lin = nn.Linear(flat, latent_dim)
            self.dec_lin = nn.Linear(latent_dim, flat)
            rev = list(channels[::-1])
            dec = []
            for i, c in enumerate(rev):
                cout = rev[i + 1] if i + 1 < len(rev) else 1
                dec.append(nn.ConvTranspose2d(c, cout, 4, stride=2, padding=1))
                if i + 1 < len(rev):
                    dec.append(act())
            self.dec_conv = nn.Sequential(*dec)

        def encode(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.enc_lin(self.enc_conv(x).flatten(1))

        def decode(self, z):
            g = self.dec_lin(z).view(z.shape[0], last, fh, fw)
            return self.dec_conv(g).squeeze(1)

        def forward(self, x):
            return self.decode(self.encode(x))

    torch.manual_seed(seed)  # seed init from the experiment seed (updated-plan §18)
    return _ConvAE()


class ConvAutoencoder(RepresentationModel):
    def __init__(
        self,
        latent_dim: int = 32,
        *,
        input_hw: tuple[int, int] = (96, 96),
        channels: tuple[int, ...] = (16, 32, 64),
        activation: str = "gelu",
        standardize: bool = False,
        trainer=None,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}")
        self.latent_dim = int(latent_dim)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.channels = tuple(int(c) for c in channels)
        self.activation = activation
        self.standardize = bool(standardize)
        self._trainer = trainer
        self.module = None
        self.scaler_: StandardScaler | None = None
        self.history_: dict | None = None

    @property
    def is_fitted(self) -> bool:
        return self.module is not None and self.history_ is not None

    def _default_trainer(self):
        if self._trainer is not None:
            return self._trainer
        from metis.training.trainer import Trainer

        return Trainer(batch_size=64, max_epochs=200, patience=25, log_every=10)

    def _apply_scaler(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler_ is None:
            return X
        flat = self.scaler_.transform(X.reshape(len(X), -1))
        return flat.reshape(X.shape).astype(np.float32)

    def fit(self, X, *, X_val=None, run=None, trainer=None) -> ConvAutoencoder:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"expected (n, H, W), got shape {X.shape}")
        if X.shape[1:] != self.input_hw:
            raise ValueError(f"input {X.shape[1:]} does not match input_hw {self.input_hw}")
        if self.standardize:
            self.scaler_ = StandardScaler().fit(X.reshape(len(X), -1))
        Xs = self._apply_scaler(X)
        Xvs = None if X_val is None else self._apply_scaler(X_val)
        trainer = trainer or self._default_trainer()
        self.module = _make_conv_net(
            self.input_hw, self.latent_dim, self.channels, self.activation, seed=trainer.seed
        )
        self.history_ = trainer.fit(self, Xs, X_val=Xvs, run=run)
        return self

    def _tensor(self, arr):
        import torch

        return torch.as_tensor(np.asarray(arr, dtype=np.float32))

    def transform(self, X):
        self._check_fitted()
        import torch

        with torch.no_grad():
            return self.module.encode(self._tensor(self._apply_scaler(X))).cpu().numpy()

    def inverse_transform(self, Z):
        self._check_fitted()
        import torch

        with torch.no_grad():
            Xs = self.module.decode(self._tensor(Z)).cpu().numpy()
        if self.scaler_ is None:
            return Xs
        flat = self.scaler_.inverse_transform(Xs.reshape(len(Xs), -1))
        return flat.reshape(Xs.shape).astype(np.float32)

    def get_params(self) -> dict:
        state = None
        if self.module is not None:
            state = {k: v.cpu().tolist() for k, v in self.module.state_dict().items()}
        return {
            "latent_dim": self.latent_dim,
            "input_hw": list(self.input_hw),
            "channels": list(self.channels),
            "activation": self.activation,
            "standardize": self.standardize,
            "scaler": None if self.scaler_ is None else self.scaler_.get_params(),
            "state_dict": state,
            "history": self.history_,
        }

    @classmethod
    def from_params(cls, params: dict) -> ConvAutoencoder:
        obj = cls(
            latent_dim=params["latent_dim"],
            input_hw=tuple(params["input_hw"]),
            channels=tuple(params["channels"]),
            activation=params["activation"],
            standardize=params["standardize"],
        )
        if params.get("scaler") is not None:
            obj.scaler_ = StandardScaler.from_params(params["scaler"])
        if params.get("state_dict") is not None:
            import torch

            obj.module = _make_conv_net(
                obj.input_hw, obj.latent_dim, obj.channels, obj.activation, seed=0
            )
            obj.module.load_state_dict(
                {k: torch.tensor(v) for k, v in params["state_dict"].items()}
            )
        obj.history_ = params.get("history")
        return obj

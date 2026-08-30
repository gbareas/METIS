"""Minimal project-level trainer for small PyTorch models (milestone I3).

Full-batch Adam with early stopping and best-state restore — enough for
the autoencoder representation study (n is tiny) and reusable by later
small models. Reproducible for a fixed seed. Optional MLflow logging via
a `metis.tracking` run handle; optional best-state checkpoint.

Deliberately not a general DL framework: no schedulers, no distributed,
no mixed precision.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trainer:
    max_epochs: int = 3000
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 300           # epochs without >min_delta improvement -> stop
    min_delta: float = 1e-7
    seed: int = 0
    device: str = "cpu"
    log_every: int = 200
    checkpoint_dir: str | Path | None = None

    def fit(self, model, X, *, run=None) -> dict:
        """Optimise `model.module` (an ``nn.Module`` whose ``forward``
        reconstructs its input) to minimise MSE on `X`. Returns a history
        dict; leaves `model.module` holding the best-seen weights."""
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        net = model.module.to(self.device)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        best_loss = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = 0
        stale = 0
        losses: list[float] = []

        for epoch in range(1, self.max_epochs + 1):
            opt.zero_grad()
            loss = loss_fn(net(X_t), X_t)
            loss.backward()
            opt.step()
            val = float(loss.detach())

            if val < best_loss - self.min_delta:
                best_loss, best_epoch, stale = val, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1

            if epoch % self.log_every == 0 or epoch == 1:
                losses.append(val)
                if run is not None:
                    run.log_metric("train_mse", val, step=epoch)

            if stale >= self.patience:
                break

        net.load_state_dict(best_state)
        history = {
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "best_train_mse": best_loss,
            "logged_losses": losses,
        }
        if self.checkpoint_dir is not None:
            ckpt_dir = Path(self.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": best_state, "epoch": best_epoch, "train_mse": best_loss},
                ckpt_dir / "best.pt",
            )
            history["checkpoint"] = str(ckpt_dir / "best.pt")
        return history

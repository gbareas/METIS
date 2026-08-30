"""Minimal project-level trainer for small PyTorch models (milestone I3;
mini-batch + validation early stopping added in I2-B / B4).

Adam with early stopping and best-state restore. Full-batch by default
(`batch_size=None`); set `batch_size` for a shuffled mini-batch loop.
Early stopping monitors the validation MSE when `fit` is given `X_val`,
otherwise the training MSE. Reproducible for a fixed seed. Optional
MLflow logging via a `metis.tracking` run handle; optional best-state
checkpoint.

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
    batch_size: int | None = None  # None -> full batch
    log_every: int = 200
    checkpoint_dir: str | Path | None = None

    def fit(self, model, X, *, X_val=None, run=None) -> dict:
        """Optimise `model.module` (an ``nn.Module`` whose ``forward``
        reconstructs its input) to minimise MSE on `X`. Returns a history
        dict; leaves `model.module` holding the best-seen weights."""
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        net = model.module.to(self.device)
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        Xv_t = (
            None if X_val is None
            else torch.as_tensor(np.asarray(X_val, dtype=np.float32), device=self.device)
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        n = X_t.shape[0]
        bs = min(self.batch_size or n, n)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        monitor_key = "val_mse" if Xv_t is not None else "train_mse"

        best_loss = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = 0
        stale = 0
        losses: list[float] = []

        for epoch in range(1, self.max_epochs + 1):
            net.train()
            order = (
                torch.randperm(n, generator=gen).to(self.device) if bs < n
                else torch.arange(n, device=self.device)
            )
            running = 0.0
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                opt.zero_grad()
                loss = loss_fn(net(X_t[idx]), X_t[idx])
                loss.backward()
                opt.step()
                running += float(loss.detach()) * len(idx)
            train_mse = running / n

            if Xv_t is not None:
                net.eval()
                with torch.no_grad():
                    monitored = float(loss_fn(net(Xv_t), Xv_t))
            else:
                monitored = train_mse

            if monitored < best_loss - self.min_delta:
                best_loss, best_epoch, stale = monitored, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                stale += 1

            if epoch % self.log_every == 0 or epoch == 1:
                losses.append(train_mse)
                if run is not None:
                    run.log_metric("train_mse", train_mse, step=epoch)
                    if Xv_t is not None:
                        run.log_metric("val_mse", monitored, step=epoch)

            if stale >= self.patience:
                break

        net.load_state_dict(best_state)
        history = {
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "monitor": monitor_key,
            f"best_{monitor_key}": best_loss,
            "logged_losses": losses,
        }
        if Xv_t is not None:  # keep a stable train-mse key too
            net.eval()
            with torch.no_grad():
                history["best_train_mse"] = float(loss_fn(net(X_t), X_t))
        if self.checkpoint_dir is not None:
            ckpt_dir = Path(self.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": best_state, "epoch": best_epoch, monitor_key: best_loss},
                ckpt_dir / "best.pt",
            )
            history["checkpoint"] = str(ckpt_dir / "best.pt")
        return history

"""I2-B / B5 — train the conv-autoencoder across every seed x latent dim.

Reads the frozen slice dataset (B2) and the frozen experiment config
(B1), then trains `metis.models.conv_autoencoder.ConvAutoencoder` for each
`(latent_dim, seed)` in the grid with `metis.training.Trainer` (Adam,
mini-batches, early stop on validation MSE, best-state checkpoint). Every
run is an MLflow run under experiment ``i2b-representation`` (tag
``phase=B5``); the training curves come from `Trainer` itself.

This milestone only *trains* — reconstruction / physical / latent
evaluation is B6-B7. The manifest it writes (``results/i2b_training.json``
by default) is the hand-off: per run it records the checkpoint path and
the best train / val MSE so B6 can reload each model.

    python scripts/i2b_train.py \
        --dataset artifacts/datasets/i2b_representation_v1_primary_u_s3_center

Smoke test:  --latent-dims 2 --seeds 0 --max-epochs 30 --limit 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from metis import tracking
from metis.data.datasets import SliceDataset

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "i2b_representation_v1.yaml"
OUT = REPO / "results" / "i2b_training.json"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch always present for this script
        return "cpu"


def _relpath(path: str | Path) -> str:
    p = Path(path)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _summary_by_k(runs: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for k in sorted({r["latent_dim"] for r in runs}):
        vals = [r for r in runs if r["latent_dim"] == k]
        vm = np.array([r["best_val_mse"] for r in vals])
        tm = np.array([r["best_train_mse"] for r in vals])
        out[str(k)] = {
            "n_seeds": len(vals),
            "val_mse_mean": float(vm.mean()), "val_mse_std": float(vm.std()),
            "train_mse_mean": float(tm.mean()), "train_mse_std": float(tm.std()),
            "epochs_run_mean": float(np.mean([r["epochs_run"] for r in vals])),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path,
                   help="slice-dataset directory built by `metis dataset build-slices`")
    p.add_argument("--config", default=CONFIG, type=Path)
    p.add_argument("--output", default=OUT, type=Path)
    p.add_argument("--checkpoint-root", type=Path, default=None,
                   help="where per-run best.pt lands (default artifacts/models/<dataset name>)")
    p.add_argument("--latent-dims", type=int, nargs="+", default=None,
                   help="override the config grid (for smoke tests)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of train/val snapshots (smoke tests only)")
    p.add_argument("--device", default=_default_device())
    p.add_argument("--save-model-json", action="store_true",
                   help="also serialise each ConvAutoencoder to model.json (heavy)")
    p.add_argument("--no-track", dest="track", action="store_false")
    args = p.parse_args(argv)

    from metis.models.conv_autoencoder import ConvAutoencoder
    from metis.training import Trainer

    cfg = yaml.safe_load(args.config.read_text())
    cand = cfg["candidate"]
    grid = list(args.latent_dims or cand["latent_dims"])
    seeds = list(args.seeds or cfg["training"]["seeds"])
    batch_size = int(cfg["training"]["batch_size"])
    channels = tuple(cand["encoder_channels"])
    activation = cand["activation"]

    ckpt_root = args.checkpoint_root or (REPO / "artifacts" / "models" / args.dataset.name)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train = SliceDataset.load(args.dataset, "train")
    val = SliceDataset.load(args.dataset, "val")
    Xtr = train.standardize()            # train-only global scaler (from metadata)
    Xva = val.standardize()
    if args.limit is not None:
        Xtr, Xva = Xtr[: args.limit], Xva[: args.limit]
    input_hw = tuple(Xtr.shape[1:])

    print(f"dataset {args.dataset.name}: train {Xtr.shape} val {Xva.shape}  "
          f"device={args.device}")
    print(f"grid k={grid}  seeds={seeds}  batch={batch_size}  "
          f"channels={channels}  act={activation}")

    runs: list[dict] = []
    for k in grid:
        for seed in seeds:
            tag = f"k{k}_seed{seed}"
            ckpt_dir = ckpt_root / tag
            trainer = Trainer(
                max_epochs=args.max_epochs, patience=args.patience, lr=args.lr,
                seed=seed, device=args.device, batch_size=batch_size,
                log_every=args.log_every, checkpoint_dir=ckpt_dir,
            )
            model = ConvAutoencoder(
                latent_dim=k, input_hw=input_hw, channels=channels,
                activation=activation, standardize=False,
            )
            params = {
                "phase": "B5", "latent_dim": k, "seed": seed,
                "batch_size": batch_size, "channels": list(channels),
                "activation": activation, "max_epochs": args.max_epochs,
                "patience": args.patience, "lr": args.lr,
                "dataset": str(args.dataset), "device": args.device,
                "n_train": len(Xtr), "n_val": len(Xva),
            }
            with tracking.run(
                "i2b-representation", run_name=f"convae-{tag}", params=params,
                tags={"phase": "B5", "model": "conv_autoencoder",
                      "latent_dim": str(k), "seed": str(seed)},
                enabled=args.track,
            ) as run:
                model.fit(Xtr, X_val=Xva, run=run, trainer=trainer)
                hist = model.history_
                run.log_metrics({
                    "best_val_mse": hist["best_val_mse"],
                    "best_train_mse": hist["best_train_mse"],
                    "best_epoch": hist["best_epoch"],
                    "epochs_run": hist["epochs_run"],
                    "compression_ratio": (input_hw[0] * input_hw[1]) / k,
                })
                run.log_artifact(hist["checkpoint"], artifact_path="checkpoint")
                if args.save_model_json:
                    from metis.data.preprocessing.base import save

                    mj = save(model, ckpt_dir / "model.json")
                    run.log_artifact(mj, artifact_path="model")
                run_id = run.run_id

            rec = {
                "latent_dim": k, "seed": seed, "mlflow_run_id": run_id,
                "checkpoint": _relpath(hist["checkpoint"]),
                "monitor": hist["monitor"],
                "best_val_mse": hist["best_val_mse"],
                "best_train_mse": hist["best_train_mse"],
                "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
            }
            runs.append(rec)
            print(f"  {tag:>12}  val_mse={rec['best_val_mse']:.5f}  "
                  f"train_mse={rec['best_train_mse']:.5f}  "
                  f"best@{rec['best_epoch']}/{rec['epochs_run']}")

    payload = {
        "dataset": str(args.dataset),
        "config": str(args.config.relative_to(REPO)),
        "field": train.field, "slice_id": train.slice_id,
        "input_hw": list(input_hw), "channels": list(channels),
        "activation": activation, "batch_size": batch_size,
        "latent_dims": grid, "seeds": seeds, "device": args.device,
        "n": {"train": len(Xtr), "val": len(Xva)},
        "scaler": train.scaler,
        "checkpoint_root": _relpath(ckpt_root),
        "runs": runs,
        "summary_by_latent_dim": _summary_by_k(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"\nsummary (val MSE, mean +/- std over {len(seeds)} seeds):")
    for k, s in payload["summary_by_latent_dim"].items():
        print(f"  k={k:>2}  {s['val_mse_mean']:.5f} +/- {s['val_mse_std']:.5f}  "
              f"(train {s['train_mse_mean']:.5f})  ~{s['epochs_run_mean']:.0f} epochs")
    print(f"\nWrote {args.output}  |  checkpoints under {ckpt_root}")
    print("Next: B6 — reload each checkpoint, run the 5.1-5.5 evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

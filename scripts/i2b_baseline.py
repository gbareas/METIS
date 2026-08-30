"""I2-B / B3 — PCA (= snapshot POD) baseline on the frozen slice dataset.

Fits PCA on the standardised training snapshots, reports reconstruction
metrics on train / val / OOD at every latent dim in the config's grid,
cross-checks the singular values against `metis.features.pod`, and picks
the headline latent dim (smallest k with >= 90% reconstructed variance).

    python scripts/i2b_baseline.py \
        --dataset artifacts/datasets/i2b_representation_v1_primary_u_s3_center

Writes results/i2b_baseline.json and (unless --no-track) an MLflow run.
Print the chosen `headline_latent_dim`; freeze it into
configs/experiments/i2b_representation_v1.yaml by hand.
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
from metis.evaluation.metrics import pointwise_metrics
from metis.evaluation.representation import choose_latent_dim, explained_variance_curve
from metis.features.pod import compute_pod
from metis.models import PCARepresentation

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "i2b_representation_v1.yaml"
OUT = REPO / "results" / "i2b_baseline.json"


def _flat_std(ds: SliceDataset) -> np.ndarray:
    return ds.standardize().reshape(len(ds), -1).astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--config", default=CONFIG, type=Path)
    p.add_argument("--output", default=OUT, type=Path)
    p.add_argument("--no-track", dest="track", action="store_false")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    grid = list(cfg["baseline"]["latent_dims"])
    threshold = 0.90

    splits = {s: SliceDataset.load(args.dataset, s) for s in ("train", "val", "ood")}
    X = {s: _flat_std(ds) for s, ds in splits.items()}

    kmax = max(grid)
    pca = PCARepresentation(latent_dim=kmax).fit(X["train"])

    # full singular spectrum of the centred training matrix (values only)
    Xc = X["train"] - X["train"].mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    cum = explained_variance_curve(sv)

    # cross-check: method-of-snapshots POD on the same centred data
    _m, _s, sv_pod, _c, _r, _e = compute_pod(Xc, energy_threshold=0.5)
    n = min(kmax, sv.size, sv_pod.size)
    sv_reldiff = float(np.max(np.abs(sv[:n] - sv_pod[:n]) / (sv_pod[:n] + 1e-30)))

    per_k = {}
    for k in grid:
        comp = pca.components_[:k]
        recon = {}
        for s in ("train", "val", "ood"):
            Z = (X[s] - pca.mean_) @ comp.T
            Xhat = Z @ comp + pca.mean_
            recon[s] = pointwise_metrics(X[s], Xhat)
        per_k[str(k)] = {
            "reconstructed_variance": float(cum[k - 1]),
            "recon": recon,
        }

    choice = choose_latent_dim(sv, grid, threshold=threshold)

    payload = {
        "dataset": str(args.dataset),
        "field": splits["train"].field,
        "slice_id": splits["train"].slice_id,
        "n": {s: len(ds) for s, ds in splits.items()},
        "singular_value_reldiff_vs_pod": sv_reldiff,
        "explained_variance_cumsum_head": cum[:kmax].tolist(),
        "per_latent_dim": per_k,
        "choice": choice,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    with tracking.run("i2b-representation", run_name="baseline-pca",
                      params={"grid": grid, "threshold": threshold,
                              "dataset": str(args.dataset)},
                      tags={"phase": "B3", "model": "pca"}, enabled=args.track) as run:
        for k in grid:
            run.log_metrics({
                f"k{k}.reconstructed_variance": per_k[str(k)]["reconstructed_variance"],
                f"k{k}.val_relative_l2": per_k[str(k)]["recon"]["val"]["relative_l2"],
                f"k{k}.ood_relative_l2": per_k[str(k)]["recon"]["ood"]["relative_l2"],
            })
        run.log_metrics({"sv_reldiff_vs_pod": sv_reldiff,
                         "headline_latent_dim": float(choice["headline_latent_dim"])})
        run.log_dict(payload, "i2b_baseline.json")

    print(f"PCA vs method-of-snapshots POD: max singular-value rel diff {sv_reldiff:.2e}")
    for k in grid:
        r = per_k[str(k)]
        print(f"  k={k:2d}  var={r['reconstructed_variance']:.3f}  "
              f"val relL2={r['recon']['val']['relative_l2']:.3f}  "
              f"ood relL2={r['recon']['ood']['relative_l2']:.3f}")
    c = choice
    tag = "" if c["reached"] else "  (threshold NOT reached in the grid)"
    print(f"\nheadline_latent_dim = {c['headline_latent_dim']}  "
          f"(reconstructed variance {c['reconstructed_variance']:.3f}, "
          f"target {c['threshold']}){tag}")
    print(f"Wrote {args.output} — freeze headline_latent_dim into {args.config.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

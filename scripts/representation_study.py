"""I2-A: does a nonlinear autoencoder latent organise the transcritical
regime any better than linear PCA on the same case-level features?

Blocks `bulk` and `rms_profile` (the R7 pressure/thermal specialists),
9 training cases + 2 OOD. Train-only standardisation, then PCA(2) vs
Autoencoder(2) over several seeds, scored with
`metis.evaluation.representation`. MLflow-logged. Writes
results/representation_study.json and prints the stop-criterion verdict.

Usage:
    python scripts/representation_study.py --data-root /path/to/dns_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from metis import tracking
from metis.config import add_data_root_args, resolve_data_root
from metis.data.datasets import build_feature_dataset
from metis.data.preprocessing import StandardScaler
from metis.data.registry import CaseRegistry
from metis.evaluation.representation import compare_to_baseline, evaluate_representation
from metis.features.regime import (
    ALL_CASE_IDS,
    OOD_CASE_IDS,
    TRAIN_CASE_IDS,
    case_grid_labels,
)
from metis.models import PCARepresentation

BLOCKS = ("bulk", "rms_profile")
SEEDS = (0, 1, 2, 3, 4)
LATENT_DIM = 2
OUT = Path(__file__).resolve().parents[1] / "results" / "representation_study.json"

_TRAIN_IDX = [ALL_CASE_IDS.index(c) for c in TRAIN_CASE_IDS]
_OOD_IDX = [ALL_CASE_IDS.index(c) for c in OOD_CASE_IDS]


def _aggregate(runs: list[dict]) -> dict:
    keys = ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc", "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc")
    return {
        **{f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys},
        **{f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys},
        "ood_nearest_Pb_Pc_centroid_by_seed": [r["ood_nearest_Pb_Pc_centroid"] for r in runs],
    }


def _study_block(block: str, X: np.ndarray, Pb, Thw) -> dict:
    from metis.models.autoencoder import Autoencoder
    from metis.training import Trainer

    scaler = StandardScaler().fit(X[_TRAIN_IDX])          # train-only, no leakage
    Xs = scaler.transform(X)
    Xs_tr, Xs_ood = Xs[_TRAIN_IDX], Xs[_OOD_IDX]
    Pb_tr = Pb[_TRAIN_IDX]
    Thw_tr = Thw[_TRAIN_IDX]

    pca = PCARepresentation(latent_dim=LATENT_DIM).fit(Xs_tr)
    pca_eval = evaluate_representation(
        pca.transform(Xs_tr), Pb_tr, Thw_tr,
        Z_ood=pca.transform(Xs_ood), ood_case_ids=list(OOD_CASE_IDS),
    )

    ae_runs = []
    for seed in SEEDS:
        ae = Autoencoder(latent_dim=LATENT_DIM, hidden=(32,), standardize=False)
        ae.fit(Xs_tr, trainer=Trainer(seed=seed, max_epochs=4000, patience=400, log_every=500))
        ae_runs.append(evaluate_representation(
            ae.transform(Xs_tr), Pb_tr, Thw_tr,
            Z_ood=ae.transform(Xs_ood), ood_case_ids=list(OOD_CASE_IDS),
        ))
    ae_agg = _aggregate(ae_runs)

    ae_mean_metrics = {k.removesuffix("_mean"): v for k, v in ae_agg.items() if k.endswith("_mean")}
    verdict = compare_to_baseline(ae_mean_metrics, pca_eval)

    return {
        "n_features": int(X.shape[1]),
        "pca": {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), **pca_eval},
        "autoencoder": {"seeds": list(SEEDS), "per_seed": ae_runs, **ae_agg},
        "autoencoder_vs_pca": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument("--output", default=OUT, type=Path)
    parser.add_argument("--no-track", dest="track", action="store_false")
    args = parser.parse_args(argv)

    registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
    ids = list(ALL_CASE_IDS)
    labels = case_grid_labels(registry, tuple(ids))
    Pb = np.array([labels[c].Pb_Pc for c in ids])
    Thw = np.array([labels[c].Thw_Tc for c in ids])

    with tracking.run(
        "representation-v1",
        params={"blocks": list(BLOCKS), "latent_dim": LATENT_DIM, "seeds": list(SEEDS),
                "data_root": str(registry.data_root)},
        tags={"kind": "study", "milestone": "I2-A"},
        enabled=args.track,
    ) as run:
        results = {}
        for block in BLOCKS:
            X = build_feature_dataset(registry, ids, block).X
            results[block] = _study_block(block, X, Pb, Thw)
            v = results[block]["autoencoder_vs_pca"]
            print(f"\n=== {block} ({results[block]['n_features']} feat) ===")
            print(f"  PCA  ARI Pb/Thw: {results[block]['pca']['ari_vs_Pb_Pc']:+.3f} / "
                  f"{results[block]['pca']['ari_vs_Thw_Tc']:+.3f}   "
                  f"OOD: {results[block]['pca']['ood_nearest_Pb_Pc_centroid']}")
            print(f"  AE   ARI Pb/Thw: {results[block]['autoencoder']['ari_vs_Pb_Pc_mean']:+.3f} / "
                  f"{results[block]['autoencoder']['ari_vs_Thw_Tc_mean']:+.3f}  "
                  f"(±{results[block]['autoencoder']['ari_vs_Pb_Pc_std']:.3f} / "
                  f"±{results[block]['autoencoder']['ari_vs_Thw_Tc_std']:.3f})")
            print(f"  AE beats PCA? {v['beats_baseline']}   improved={v['improved']} "
                  f"regressed={v['regressed']}")
            run.log_metrics({
                f"{block}.pca.ari_vs_Pb_Pc": results[block]["pca"]["ari_vs_Pb_Pc"],
                f"{block}.pca.ari_vs_Thw_Tc": results[block]["pca"]["ari_vs_Thw_Tc"],
                f"{block}.ae.ari_vs_Pb_Pc_mean": results[block]["autoencoder"]["ari_vs_Pb_Pc_mean"],
                f"{block}.ae.ari_vs_Thw_Tc_mean": results[block]["autoencoder"]["ari_vs_Thw_Tc_mean"],
                f"{block}.ae_beats_pca": float(v["beats_baseline"]),
            })

        any_beats = any(r["autoencoder_vs_pca"]["beats_baseline"] for r in results.values())
        verdict = (
            "AUTOENCODER ADDS SIGNAL — continue (I2-B slice fields)"
            if any_beats else
            "STOP CRITERION MET — AE latent does not beat PCA on any block; "
            "case-level representation learning is parked"
        )
        print(f"\n{verdict}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"blocks": results, "any_block_ae_beats_pca": any_beats, "verdict": verdict}
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")
        run.log_metric("any_block_ae_beats_pca", float(any_beats))
        run.log_dict(payload, "representation_study.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Per-block ablation for the regime-discovery reference task
(FINDINGS.md §2's "next" step).

The rich feature set (scripts/run_regime_discovery.py) concatenates four
feature blocks and standardizes them per-feature before PCA, which turned
out to hand implicit "voting power" to whichever axis happens to modulate
more individual features rather than whichever axis is physically more
important. This script runs each block from metis.features.regime in
isolation — bulk, mean_profile, rms_profile, pod — through the exact same
H1/H2/H3 pipeline, to find out which block actually carries the pressure
vs. thermal signal before attempting any block-weighted combination.

Usage: python scripts/run_regime_ablation.py
"""
from __future__ import annotations

import json
from pathlib import Path

from metis.evaluation.regime import evaluate_regime_discovery
from metis.features.regime import (
    ALL_CASE_IDS,
    FEATURE_BLOCK_NAMES,
    build_feature_matrix_block,
    case_grid_labels,
)

DATA_ROOT = Path("/home/brinkman/Documents/PostDoc_phase/data")
OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    labels = case_grid_labels(DATA_ROOT, ALL_CASE_IDS)

    results = {}
    for block in FEATURE_BLOCK_NAMES:
        X, names = build_feature_matrix_block(ALL_CASE_IDS, DATA_ROOT, block)
        results[block] = {"feature_names": names, **evaluate_regime_discovery(X, labels)}

    header = f"{'block':14s} {'n_feat':>7s} {'ARI_Pb':>8s} {'ARI_Thw':>8s} {'LOCO_Pb':>8s} {'LOCO_Thw':>9s}"
    print(header)
    print("-" * len(header))
    for block, r in results.items():
        print(
            f"{block:14s} {r['n_features']:7d} {r['ari_vs_Pb_Pc']:8.3f} "
            f"{r['ari_vs_Thw_Tc']:8.3f} {r['loco_accuracy_Pb_Pc']:8.3f} "
            f"{r['loco_accuracy_Thw_Tc']:9.3f}"
        )
        print(f"    OOD nearest Pb_Pc centroid: {r['ood_nearest_Pb_Pc_centroid']}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "regime_discovery_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_DIR / 'regime_discovery_ablation.json'}")


if __name__ == "__main__":
    main()

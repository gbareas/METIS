"""Block-wise (MFA-style) feature combination for the regime-discovery
reference task (FINDINGS.md §3's "next" step).

FINDINGS.md §3's per-block ablation found the pressure signal (H1) in
`bulk` and the thermal signal (H2) in `rms_profile`, with `mean_profile`
uninformative and `pod` inconsistent. Naive concatenation
(scripts/run_regime_discovery.py's "rich" set) diluted `bulk`'s pressure
signal because per-feature standardization gives every one of
`mean_profile`'s 384 features equal weight to `bulk`'s 7.

This script combines `bulk` + `rms_profile` (and, separately,
`bulk` + `rms_profile` + `pod`, to actually test `pod`'s provisional
status rather than guess) via `metis.evaluation.regime.combine_blocks_mfa`
— each block standardized then divided by its own leading singular value,
so blocks compete on genuine structure rather than raw feature count —
and compares against naive concatenation of the same blocks as a control.

Usage: python scripts/run_regime_blockwise.py --data-root /path/to/dns_data
       (or set METIS_DATA_ROOT / configs/default.yaml — see metis.config)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metis.config import add_data_root_args, data_root_from_args
from metis.evaluation.regime import combine_blocks_mfa, evaluate_regime_discovery
from metis.features.regime import (
    ALL_CASE_IDS,
    build_feature_matrix_block,
    case_grid_labels,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument(
        "--output", default=OUT_DIR / "regime_discovery_blockwise.json", type=Path,
        help="where to write the results JSON",
    )
    args = parser.parse_args(argv)
    data_root = data_root_from_args(args)

    labels = case_grid_labels(data_root, ALL_CASE_IDS)

    block_matrices = {
        block: build_feature_matrix_block(ALL_CASE_IDS, data_root, block)[0]
        for block in ("bulk", "rms_profile", "pod")
    }

    combos = {
        "bulk+rms_profile": ("bulk", "rms_profile"),
        "bulk+rms_profile+pod": ("bulk", "rms_profile", "pod"),
    }

    results = {}
    for combo_name, block_names in combos.items():
        blocks = {b: block_matrices[b] for b in block_names}
        naive_X = np.concatenate([blocks[b] for b in block_names], axis=1)
        mfa_X = combine_blocks_mfa(blocks)

        results[combo_name] = {
            "naive": evaluate_regime_discovery(naive_X, labels),
            "mfa": evaluate_regime_discovery(mfa_X, labels, standardize_input=False),
        }

    header = f"{'combo':24s} {'method':6s} {'ARI_Pb':>8s} {'ARI_Thw':>8s} {'LOCO_Pb':>8s} {'LOCO_Thw':>9s}"
    print(header)
    print("-" * len(header))
    for combo_name, methods in results.items():
        for method_name, r in methods.items():
            print(
                f"{combo_name:24s} {method_name:6s} {r['ari_vs_Pb_Pc']:8.3f} "
                f"{r['ari_vs_Thw_Tc']:8.3f} {r['loco_accuracy_Pb_Pc']:8.3f} "
                f"{r['loco_accuracy_Thw_Tc']:9.3f}"
            )
            print(f"    OOD nearest Pb_Pc centroid: {r['ood_nearest_Pb_Pc_centroid']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

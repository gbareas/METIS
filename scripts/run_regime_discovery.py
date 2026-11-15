"""Run the regime-discovery reference task (research_protocol.md) against
real DNS output, for both feature sets in metis.features.regime:

    compact - 14 features: bulk groups + 2 RMS-peak scalars + 1-slice POD.
    rich    - ~680 features: full mean/RMS profiles + multi-slice,
              multi-field POD (FINDINGS.md §1's proposed enrichment).

See metis.evaluation.regime.evaluate_regime_discovery for the shared
H1/H2/H3 evaluation (ARI, LOCO nearest-centroid accuracy, OOD nearest
centroid) both feature sets are judged by.

Usage: python scripts/run_regime_discovery.py --data-root /path/to/dns_data
       (or set METIS_DATA_ROOT / configs/default.yaml — see metis.config)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metis.config import add_data_root_args, data_root_from_args
from metis.evaluation.regime import evaluate_regime_discovery
from metis.features.regime import (
    ALL_CASE_IDS,
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_matrix_rich,
    case_grid_labels,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "results"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument(
        "--output", default=OUT_DIR / "regime_discovery.json", type=Path,
        help="where to write the results JSON",
    )
    args = parser.parse_args(argv)
    data_root = data_root_from_args(args)

    labels = case_grid_labels(data_root, ALL_CASE_IDS)

    X_compact = build_feature_matrix(ALL_CASE_IDS, data_root)
    X_rich, rich_names = build_feature_matrix_rich(ALL_CASE_IDS, data_root)

    results = {
        "case_ids": list(ALL_CASE_IDS),
        "compact": {"feature_names": FEATURE_NAMES, **evaluate_regime_discovery(X_compact, labels)},
        "rich": {"feature_names": rich_names, **evaluate_regime_discovery(X_rich, labels)},
    }

    summary = {k: v for k, v in results.items() if k != "case_ids"}
    for name, r in summary.items():
        print(f"\n=== {name} ({r['n_features']} features) ===")
        print(f"  ARI vs Pb_Pc:   {r['ari_vs_Pb_Pc']:.3f}")
        print(f"  ARI vs Thw_Tc:  {r['ari_vs_Thw_Tc']:.3f}")
        print(f"  LOCO Pb_Pc:     {r['loco_accuracy_Pb_Pc']:.3f}")
        print(f"  LOCO Thw_Tc:    {r['loco_accuracy_Thw_Tc']:.3f}")
        print(f"  OOD nearest Pb_Pc centroid: {r['ood_nearest_Pb_Pc_centroid']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

"""Build (and cache) a Level-1 feature dataset artifact (milestone R5).

Usage:
    python scripts/build_feature_dataset.py --feature-set compact \
        --out artifacts/datasets/regime_compact_v1 --data-root /path/to/dns_data

Cases default to the regime train+OOD set (case01-09, case10, case15).
Re-running is a no-op if the artifact is already up to date; pass
--rebuild to force recomputation.
"""
from __future__ import annotations

import argparse

from metis.config import add_data_root_args, resolve_data_root
from metis.data.datasets import FEATURE_SETS, build_feature_dataset
from metis.data.registry import CaseRegistry
from metis.features.regime import ALL_CASE_IDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument("--feature-set", default="compact", choices=FEATURE_SETS)
    parser.add_argument(
        "--cases", default=None,
        help="comma-separated case ids (default: the regime train+OOD set)",
    )
    parser.add_argument("--out", required=True, help="artifact directory")
    parser.add_argument("--rebuild", action="store_true", help="ignore any cached artifact")
    args = parser.parse_args(argv)

    data_root = resolve_data_root(cli_value=args.data_root, config_path=args.config)
    registry = CaseRegistry(data_root)
    case_ids = args.cases.split(",") if args.cases else list(ALL_CASE_IDS)

    dataset = build_feature_dataset(
        registry, case_ids, args.feature_set, out_dir=args.out, rebuild=args.rebuild,
    )
    p = dataset.provenance
    print(
        f"{args.feature_set}: {p['n_cases']} cases x {p['n_features']} features "
        f"-> {args.out}\n  fingerprint {p['fingerprint']}  code {p['code_version'][:12]}"
    )


if __name__ == "__main__":
    main()

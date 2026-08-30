"""Run a standard analysis on one DNS case (milestone R6).

Usage:
    python scripts/analyze.py physics case01 --data-root /path/to/dns_data
    python scripts/analyze.py spectra case01 --slice s3_center --field u --axis x
    python scripts/analyze.py pod case01 --slice s2_max_u --field u
    python scripts/analyze.py regime-features case01 --feature-set compact

Add --out DIR to write outputs.json + arrays.npz; --no-validate to skip
the data-validation pass.
"""
from __future__ import annotations

import argparse

from metis.analysis import ANALYSES, run_analysis
from metis.config import add_data_root_args, resolve_data_root
from metis.data.registry import CaseRegistry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=[a.replace("_", "-") for a in ANALYSES])
    parser.add_argument("case")
    add_data_root_args(parser)
    parser.add_argument("--slice", dest="slice_id", default=None)
    parser.add_argument("--field", default=None)
    parser.add_argument("--axis", default=None, choices=["x", "z"])
    parser.add_argument("--feature-set", dest="feature_set", default=None)
    parser.add_argument("--energy-threshold", dest="energy_threshold", type=float, default=None)
    parser.add_argument("--out", default=None, help="directory for outputs.json + arrays.npz")
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    args = parser.parse_args(argv)

    registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
    options = {
        k: getattr(args, k)
        for k in ("slice_id", "field", "axis", "feature_set", "energy_threshold")
        if getattr(args, k) is not None
    }
    result = run_analysis(
        registry, args.analysis.replace("-", "_"), args.case,
        validate=args.validate, **options,
    )
    print(result.summary())
    for name, value in result.outputs.items():
        if not isinstance(value, (list, dict)):
            print(f"  {name}: {value}")
    if result.validation and not result.validation["ok"]:
        for issue in result.validation["issues"]:
            print(f"  ! {issue['severity']} {issue['check']}: {issue['message']}")
    if args.out:
        result.save(args.out)
        print(f"  wrote {args.out}/")


if __name__ == "__main__":
    main()

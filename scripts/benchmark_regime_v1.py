"""Reproduce the frozen `regime-v1` benchmark against real DNS data
(milestone R7.3 — the `metis benchmark regime-v1` command).

Usage:
    python scripts/benchmark_regime_v1.py --data-root /path/to/dns_data

Builds the four block feature matrices for case01-09/10/15 (cached via
metis.data.datasets), runs the settled regime-discovery evaluation, scores
the frozen checks, and writes results/regime_v1.json. Exits non-zero if
any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metis.config import add_data_root_args, resolve_data_root
from metis.data.registry import CaseRegistry
from metis.evaluation.benchmark import load_config, run_regime_v1_from_registry

OUT = Path(__file__).resolve().parents[1] / "results" / "regime_v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument("--benchmark-config", dest="benchmark_config", default=None,
                        help="benchmark spec (default: configs/benchmarks/regime_discovery_v1.yaml)")
    parser.add_argument("--output", default=OUT, type=Path)
    args = parser.parse_args(argv)

    registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
    config = load_config(args.benchmark_config)
    result = run_regime_v1_from_registry(registry, config)

    print(result.summary())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "name": result.name,
        "passed": result.passed,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks],
        "blocks": result.blocks,
        "combined": result.combined,
        "provenance": result.provenance,
    }, indent=2))
    print(f"\nWrote {args.output}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

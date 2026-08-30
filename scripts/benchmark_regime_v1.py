"""Reproduce the frozen `regime-v1` benchmark against real DNS data
(milestone R7.3 — the `metis benchmark regime-v1` command).

Usage:
    python scripts/benchmark_regime_v1.py --data-root /path/to/dns_data

Builds the four block feature matrices for case01-09/10/15 (cached via
metis.data.datasets), runs the settled regime-discovery evaluation, scores
the frozen checks, and writes results/regime_v1.json. Exits non-zero if
any check fails. Logs the run to MLflow (local mlruns/) when the `ml`
extra is installed; pass --no-track to skip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metis import tracking
from metis.config import add_data_root_args, resolve_data_root
from metis.data.registry import CaseRegistry
from metis.evaluation.benchmark import load_config, run_regime_v1_from_registry

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "regime_v1.json"
BENCHMARK_CONFIG = REPO / "configs" / "benchmarks" / "regime_discovery_v1.yaml"


def _run_metrics(result) -> dict[str, float]:
    metrics: dict[str, float] = {"passed": float(result.passed)}
    for block, ev in result.blocks.items():
        for k in ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc", "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc"):
            metrics[f"{block}.{k}"] = float(ev[k])
    for combo, ev in result.combined.items():
        metrics[f"{combo}.ari_vs_Pb_Pc"] = float(ev["ari_vs_Pb_Pc"])
        metrics[f"{combo}.ari_vs_Thw_Tc"] = float(ev["ari_vs_Thw_Tc"])
    for check in result.checks:
        metrics[f"check.{check.name}"] = float(check.passed)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_args(parser)
    parser.add_argument("--benchmark-config", dest="benchmark_config", default=None,
                        help="benchmark spec (default: configs/benchmarks/regime_discovery_v1.yaml)")
    parser.add_argument("--output", default=OUT, type=Path)
    parser.add_argument("--no-track", dest="track", action="store_false",
                        help="do not log this run to MLflow")
    args = parser.parse_args(argv)

    registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
    config_path = args.benchmark_config or BENCHMARK_CONFIG
    config = load_config(config_path)

    with tracking.run(
        "regime-v1",
        params={"benchmark": config, "data_root": str(registry.data_root)},
        tags={"kind": "benchmark"},
        enabled=args.track,
    ) as run_handle:
        result = run_regime_v1_from_registry(registry, config)
        print(result.summary())

        payload = {
            "name": result.name,
            "passed": result.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks
            ],
            "blocks": result.blocks,
            "combined": result.combined,
            "provenance": result.provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")

        run_handle.log_metrics(_run_metrics(result))
        run_handle.log_dict(payload, "regime_v1.json")
        if Path(config_path).exists():
            run_handle.log_artifact(config_path, artifact_path="config")
        if run_handle.active:
            print(f"MLflow run: {run_handle.run_id}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

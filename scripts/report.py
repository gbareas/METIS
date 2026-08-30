"""Generate a report from an experiment result (milestone I6).

    python scripts/report.py regime-v1        --from results/regime_v1.json
    python scripts/report.py representation   --from results/representation_study.json
    python scripts/report.py regime-v1        --run-id <mlflow_run_id>
    python scripts/report.py physics case01   --data-root /path/to/dns_data

Writes reports/<name>/{summary.md, metrics.json, figures/} (figures need
the `report` extra). --run-id pulls the result JSON from an MLflow run's
artifacts (needs the `ml` extra).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metis.config import add_data_root_args, resolve_data_root
from metis.reporting.generators import GENERATORS, physics_report

REPORTS = Path(__file__).resolve().parents[1] / "reports"
_ARTIFACT_NAME = {"regime-v1": "regime_v1.json", "representation": "representation_study.json"}


def _load_from_run(run_id: str, kind: str) -> dict:
    import mlflow

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=_ARTIFACT_NAME[kind]
    )
    return json.loads(Path(local).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=[*GENERATORS, "physics"])
    parser.add_argument("case", nargs="?", help="case id (physics report only)")
    parser.add_argument("--from", dest="from_json", type=Path, default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--out", type=Path, default=None)
    add_data_root_args(parser)
    args = parser.parse_args(argv)

    if args.kind == "physics":
        if not args.case:
            parser.error("physics report needs a case id")
        from metis.analysis import run_analysis
        from metis.data.registry import CaseRegistry

        registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
        report = physics_report(run_analysis(registry, "physics", args.case))
        out = args.out or REPORTS / f"physics-{args.case}"
    else:
        if args.run_id:
            data = _load_from_run(args.run_id, args.kind)
        elif args.from_json:
            data = json.loads(args.from_json.read_text())
        else:
            parser.error("give --from <json> or --run-id <id>")
        report = GENERATORS[args.kind](data)
        out = args.out or REPORTS / args.kind

    written = report.write(out)
    print(f"Wrote {written}/summary.md ({len(report.metrics)} metrics, "
          f"{len(report.figures)} figure(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())

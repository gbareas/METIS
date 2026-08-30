"""`metis` — one user-facing command over the framework (milestone I7).

    metis cases list
    metis case validate case01
    metis analyze physics case01
    metis dataset build --feature-set compact --out artifacts/datasets/compact_v1
    metis benchmark regime-v1
    metis report regime-v1 --from results/regime_v1.json

Every subcommand wraps a `metis.*` library call; `scripts/*.py` are thin
shims over this. Data-root resolution follows `metis.config`
(--data-root > --config's data.root > $METIS_DATA_ROOT).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from metis import __version__
from metis.config import add_data_root_args, resolve_data_root
from metis.data.validation import ValidationError


def _die(msg: str) -> int:
    print(f"metis: error: {msg}", file=sys.stderr)
    return 2


def _registry(args):
    from metis.data.registry import CaseRegistry

    return CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))


# --------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------- #
def _cmd_cases_list(args) -> int:
    reg = _registry(args)
    ids = reg.case_ids()
    if not ids:
        print(f"no cases found under {reg.processed_root}")
        return 0
    print(f"{'case':10s} {'Pb_Pc':>7s} {'Thw_Tc':>7s} {'Tcw_Tc':>7s}  raw slices")
    for cid in ids:
        d = reg[cid]
        print(f"{cid:10s} {d.Pb_Pc:7.3g} {d.Thw_Tc:7.3g} {d.Tcw_Tc:7.3g}  "
              f"{'y' if d.has_raw else '-'}   {','.join(d.available_slices()) or '-'}")
    return 0


def _cmd_case_validate(args) -> int:
    from metis.data.validation import validate_case

    reg = _registry(args)
    if args.case not in reg:
        return _die(f"unknown case {args.case!r}; known: {reg.case_ids() or 'none'}")
    slices = args.slices.split(",") if args.slices else None
    report = validate_case(reg[args.case])
    if slices:
        from metis.data.validation import validate_slice_case

        for sid in slices:
            report.extend(validate_slice_case(reg[args.case].load_slice(sid)))
    print(report.summary())
    return 0 if report.ok else 1


# --------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------- #
_ANALYSIS_OPTS = ("slice_id", "field", "axis", "feature_set", "energy_threshold")


def _cmd_analyze(args) -> int:
    from metis.analysis import run_analysis

    reg = _registry(args)
    if args.case not in reg:
        return _die(f"unknown case {args.case!r}; known: {reg.case_ids() or 'none'}")
    opts = {k: getattr(args, k) for k in _ANALYSIS_OPTS if getattr(args, k) is not None}
    result = run_analysis(reg, args.kind.replace("-", "_"), args.case,
                          validate=args.validate, strict=not args.allow_invalid, **opts)
    print(result.summary())
    for k, v in result.outputs.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")
    if result.validation and not result.validation["ok"]:
        for i in result.validation["issues"]:
            print(f"  ! {i['severity']} {i['check']}: {i['message']}")
    if args.out:
        result.save(args.out)
        print(f"  wrote {args.out}/")
    return 0


# --------------------------------------------------------------------- #
# dataset build
# --------------------------------------------------------------------- #
def _cmd_dataset_build(args) -> int:
    from metis.data.datasets import build_feature_dataset
    from metis.features.regime import ALL_CASE_IDS

    reg = _registry(args)
    case_ids = args.cases.split(",") if args.cases else list(ALL_CASE_IDS)
    ds = build_feature_dataset(reg, case_ids, args.feature_set,
                               out_dir=args.out, rebuild=args.rebuild)
    p = ds.provenance
    print(f"{args.feature_set}: {p['n_cases']} cases x {p['n_features']} features -> {args.out}\n"
          f"  fingerprint {p['fingerprint']}  code {p['code_version'][:12]}")
    if args.register:
        from metis.registry import ArtifactRegistry

        entry = ArtifactRegistry(args.registry_root).register(
            args.register, "dataset", path=args.out, scope=args.scope,
            metrics={"fingerprint": p["fingerprint"], "n_cases": p["n_cases"],
                     "n_features": p["n_features"], "feature_set": args.feature_set},
        )
        print(f"  registered as {entry.id!r} ({entry.status})")
    return 0


def _cmd_dataset_build_slices(args) -> int:
    import yaml

    from metis.data.datasets import build_slice_dataset, normalize_split_config

    reg = _registry(args)
    cfg = yaml.safe_load(Path(args.experiment_config).read_text())
    field = args.field or cfg["data"]["field"]
    slice_id = args.slice_id or cfg["data"]["slice_id"]
    splits = normalize_split_config(cfg["splits"][args.split])
    out = args.out or (
        Path("artifacts") / "datasets" / f"{cfg['name']}_{args.split}_{field}_{slice_id}"
    )
    ds = build_slice_dataset(reg, field, slice_id, splits, out_dir=out, rebuild=args.rebuild)
    n = ds["train"].provenance["n"]
    print(f"{field} @ {slice_id} [{args.split}] -> {out}")
    print(f"  train={n['train']}  val={n['val']}  ood={n['ood']}  "
          f"grid={list(ds['train'].X.shape[1:])}  fp={ds['train'].provenance['fingerprint']}")
    return 0


# --------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------- #
def _artifact_registry(args):
    from metis.registry import ArtifactRegistry

    return ArtifactRegistry(args.registry_root)


def _cmd_registry_list(args) -> int:
    entries = _artifact_registry(args).list(kind=args.kind, status=args.status)
    if not entries:
        print("(registry empty)")
        return 0
    print(f"{'id':28s} {'kind':8s} {'status':13s} {'run_id':10s} scope")
    for e in entries:
        print(f"{e.id:28s} {e.kind:8s} {e.status:13s} "
              f"{(e.run_id or '-')[:10]:10s} {e.scope or '-'}")
    return 0


def _cmd_registry_show(args) -> int:
    print(json.dumps(asdict(_artifact_registry(args)[args.id]), indent=2))
    return 0


def _cmd_registry_add(args) -> int:
    metrics = {}
    for kv in args.metric or []:
        k, _, v = kv.partition("=")
        try:
            metrics[k] = float(v)
        except ValueError:
            metrics[k] = v
    entry = _artifact_registry(args).register(
        args.id, args.kind, status=args.status, run_id=args.run_id,
        dataset_id=args.dataset_id, path=args.path, scope=args.scope, metrics=metrics,
    )
    print(f"registered {entry.id!r} ({entry.kind}, {entry.status})")
    return 0


def _cmd_registry_status(args) -> int:
    entry = _artifact_registry(args).set_status(args.id, args.status_value)
    print(f"{entry.id!r} -> {entry.status}")
    return 0


# --------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------- #
def _cmd_benchmark(args) -> int:
    from metis import tracking
    from metis.evaluation.benchmark import load_config, run_regime_v1_from_registry

    reg = _registry(args)
    config = load_config(args.benchmark_config)
    with tracking.run("regime-v1", params={"benchmark": config, "data_root": str(reg.data_root)},
                      tags={"kind": "benchmark"}, enabled=args.track) as run:
        result = run_regime_v1_from_registry(
            reg, config,
            cache_dir=None if args.no_cache else args.cache_dir,
            rebuild=args.rebuild,
        )
        print(result.summary())
        if "cache" in result.provenance:
            print("  feature cache: " + ", ".join(
                f"{b}={s}" for b, s in result.provenance["cache"].items()))
        payload = {
            "name": result.name, "passed": result.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in result.checks],
            "blocks": result.blocks, "combined": result.combined,
            "provenance": result.provenance,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")

        metrics = {"passed": float(result.passed)}
        for b, ev in result.blocks.items():
            for m in ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc",
                      "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc"):
                metrics[f"{b}.{m}"] = float(ev[m])
        for c in result.checks:
            metrics[f"check.{c.name}"] = float(c.passed)
        run.log_metrics(metrics)
        run.log_dict(payload, "regime_v1.json")
        if run.active:
            print(f"MLflow run: {run.run_id}")
    return 0 if result.passed else 1


# --------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------- #
_I2B_BUNDLE_FILES = {
    "baseline": "i2b_baseline.json", "training": "i2b_training.json",
    "evaluation": "i2b_evaluation.json", "decision": "i2b_decision.json",
    "modal_compare": "i2b_modal_compare.json",
}


def _load_i2b_bundle(path: Path) -> dict | None:
    root = path if path.is_dir() else path.parent
    bundle = {
        key: json.loads((root / fn).read_text())
        for key, fn in _I2B_BUNDLE_FILES.items() if (root / fn).exists()
    }
    missing = {"evaluation", "decision"} - bundle.keys()
    if missing:
        _die(f"i2b-representation report needs {sorted(missing)} under {root}")
        return None
    return bundle


def _cmd_report(args) -> int:
    from metis.reporting.generators import GENERATORS, physics_report

    reports_dir = Path("reports")
    if args.kind == "physics":
        if not args.target:
            return _die("`metis report physics` needs a case id")
        from metis.analysis import run_analysis

        reg = _registry(args)
        report = physics_report(run_analysis(reg, "physics", args.target))
        out = args.out or reports_dir / f"physics-{args.target}"
    elif args.kind == "i2b-representation":
        if not args.from_json:
            return _die("metis report i2b-representation needs --from results/ "
                        "(a directory, or any of its i2b_*.json)")
        data = _load_i2b_bundle(args.from_json)
        if data is None:
            return 2
        report = GENERATORS[args.kind](data)
        out = args.out or reports_dir / args.kind
    else:
        if args.run_id:
            import mlflow

            name = {"regime-v1": "regime_v1.json", "representation": "representation_study.json"}
            local = mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path=name[args.kind])
            data = json.loads(Path(local).read_text())
        elif args.from_json:
            data = json.loads(args.from_json.read_text())
        else:
            return _die("give --from <json> or --run-id <id>")
        report = GENERATORS[args.kind](data)
        out = args.out or reports_dir / args.kind

    written = report.write(out)
    print(f"Wrote {written}/summary.md ({len(report.metrics)} metrics, "
          f"{len(report.figures)} figure(s))")
    return 0


# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metis", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"metis {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # cases
    p_cases = sub.add_parser("cases", help="inspect the DNS case registry").add_subparsers(
        dest="cases_command", metavar="<subcommand>")
    pc = p_cases.add_parser("list", help="list registered cases")
    add_data_root_args(pc)
    pc.set_defaults(func=_cmd_cases_list)

    # case validate
    p_case = sub.add_parser("case", help="operate on one case").add_subparsers(
        dest="case_command", metavar="<subcommand>")
    pv = p_case.add_parser("validate", help="validate a case's data")
    pv.add_argument("case")
    pv.add_argument("--slices", default=None, help="comma-separated slice ids to also check")
    add_data_root_args(pv)
    pv.set_defaults(func=_cmd_case_validate)

    # analyze
    pa = sub.add_parser("analyze", help="run a standard analysis on a case")
    pa.add_argument("kind", choices=["physics", "spectra", "pod", "regime-features"])
    pa.add_argument("case")
    pa.add_argument("--slice", dest="slice_id", default=None)
    pa.add_argument("--field", default=None)
    pa.add_argument("--axis", default=None, choices=["x", "z"])
    pa.add_argument("--feature-set", dest="feature_set", default=None)
    pa.add_argument("--energy-threshold", dest="energy_threshold", type=float, default=None)
    pa.add_argument("--out", default=None, help="directory for outputs.json + arrays.npz")
    pa.add_argument("--no-validate", dest="validate", action="store_false",
                    help="skip the data-validation pass entirely")
    pa.add_argument("--allow-invalid", dest="allow_invalid", action="store_true",
                    help="produce a result even if validation reports errors "
                         "(recorded in provenance)")
    add_data_root_args(pa)
    pa.set_defaults(func=_cmd_analyze)

    # dataset build
    p_ds = sub.add_parser("dataset", help="feature dataset artifacts").add_subparsers(
        dest="dataset_command", metavar="<subcommand>")
    pb = p_ds.add_parser("build", help="build (and cache) a Level-1 feature dataset")
    pb.add_argument("--feature-set", dest="feature_set", default="compact",
                    choices=["compact", "rich", "bulk", "mean_profile", "rms_profile", "pod"])
    pb.add_argument("--cases", default=None, help="comma-separated (default: regime train+OOD)")
    pb.add_argument("--out", required=True)
    pb.add_argument("--rebuild", action="store_true")
    pb.add_argument("--register", default=None, metavar="ID",
                    help="also record this dataset in the artifact registry")
    pb.add_argument("--scope", default=None, help="documented reuse scope (for --register)")
    pb.add_argument("--registry-root", dest="registry_root", default="artifacts")
    add_data_root_args(pb)
    pb.set_defaults(func=_cmd_dataset_build)

    pbs = p_ds.add_parser("build-slices",
                          help="build (and cache) a Level-2 slice-snapshot dataset")
    pbs.add_argument("--experiment-config", dest="experiment_config", required=True,
                     help="e.g. configs/experiments/i2b_representation_v1.yaml")
    pbs.add_argument("--split", default="primary", help="which splits: block in the config")
    pbs.add_argument("--field", default=None, help="override the config's data.field")
    pbs.add_argument("--slice", dest="slice_id", default=None,
                     help="override the config's data.slice_id")
    pbs.add_argument("--out", default=None)
    pbs.add_argument("--rebuild", action="store_true")
    add_data_root_args(pbs)
    pbs.set_defaults(func=_cmd_dataset_build_slices)

    # registry
    p_reg = sub.add_parser("registry", help="the artifact (dataset/model) registry")
    p_reg.add_argument("--registry-root", dest="registry_root", default="artifacts")
    reg_sub = p_reg.add_subparsers(dest="registry_command", metavar="<subcommand>")

    rl = reg_sub.add_parser("list", help="list registered artifacts")
    rl.add_argument("--kind", choices=["dataset", "model", "report"], default=None)
    rl.add_argument("--status", choices=["experimental", "validated", "deprecated"], default=None)
    rl.set_defaults(func=_cmd_registry_list)

    rs = reg_sub.add_parser("show", help="show one artifact entry")
    rs.add_argument("id")
    rs.set_defaults(func=_cmd_registry_show)

    ra = reg_sub.add_parser("add", help="register an artifact")
    ra.add_argument("id")
    ra.add_argument("--kind", required=True, choices=["dataset", "model", "report"])
    ra.add_argument("--status", default="experimental",
                    choices=["experimental", "validated", "deprecated"])
    ra.add_argument("--run-id", dest="run_id", default=None)
    ra.add_argument("--dataset-id", dest="dataset_id", default=None)
    ra.add_argument("--path", default=None)
    ra.add_argument("--scope", default=None)
    ra.add_argument("--metric", action="append", metavar="KEY=VALUE",
                    help="repeatable; recorded on the entry")
    ra.set_defaults(func=_cmd_registry_add)

    for name, val in (("promote", "validated"), ("deprecate", "deprecated")):
        rp = reg_sub.add_parser(name, help=f"set an artifact's status to {val}")
        rp.add_argument("id")
        rp.set_defaults(func=_cmd_registry_status, status_value=val)
    rss = reg_sub.add_parser("set-status", help="set an artifact's status")
    rss.add_argument("id")
    rss.add_argument("status_value", choices=["experimental", "validated", "deprecated"])
    rss.set_defaults(func=_cmd_registry_status, status_value=None)

    # benchmark
    pbm = sub.add_parser("benchmark", help="run a frozen benchmark")
    pbm.add_argument("name", choices=["regime-v1"])
    pbm.add_argument("--benchmark-config", dest="benchmark_config", default=None)
    pbm.add_argument("--output", type=Path,
                     default=Path("results") / "regime_v1.json")
    pbm.add_argument("--no-track", dest="track", action="store_false")
    pbm.add_argument("--cache-dir", dest="cache_dir",
                     default=str(Path("artifacts") / "datasets" / "regime-v1"),
                     help="where the feature blocks are cached")
    pbm.add_argument("--no-cache", dest="no_cache", action="store_true",
                     help="always rebuild feature blocks from raw DNS")
    pbm.add_argument("--rebuild", action="store_true",
                     help="ignore any cached feature blocks (rebuild + rewrite)")
    add_data_root_args(pbm)
    pbm.set_defaults(func=_cmd_benchmark)

    # report
    pr = sub.add_parser("report", help="build a report from an experiment result")
    pr.add_argument("kind", choices=["regime-v1", "representation", "i2b-representation", "physics"])
    pr.add_argument("target", nargs="?", help="case id (physics only)")
    pr.add_argument("--from", dest="from_json", type=Path, default=None)
    pr.add_argument("--run-id", dest="run_id", default=None)
    pr.add_argument("--out", type=Path, default=None)
    add_data_root_args(pr)
    pr.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args) or 0
    except ValidationError as exc:
        print(exc.report.summary(), file=sys.stderr)
        return _die("input failed validation (pass --allow-invalid to override)")
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        return _die(str(exc))


if __name__ == "__main__":
    sys.exit(main())

"""`metis demo` — a self-contained tour that runs anywhere, in about a minute.

No group DNS data, no GPU, no model checkpoint and no API key needed. It
runs three things and writes everything to one output folder:

1. **Pipeline on synthetic data.** Generates a small DNS-shaped HDF5 case
   (`metis.testing.mock_dns`, random fields, *not* physics), then runs
   ingest -> validate -> physics analysis -> report on it. This shows the
   framework end to end; the numbers themselves are meaningless.
2. **The frozen `regime-v1` benchmark on real-derived features.** Re-runs
   the benchmark from the committed ~60 KB feature sample
   (`tests/data/regime_v1/block_features.npz`, extracted once from the 11
   real DNS cases) and checks all 7 frozen checks still pass.
3. **The router's OOD confidence policy.** Scores the curated router
   scenarios with the rule-based diagnostic (`metis.router.confidence`).
   Only the routing decision runs here; the surrogate checkpoint and the
   DNS database it would dispatch to are not part of the public repo.

It finishes with a two-panel summary figure (`summary.png`).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = REPO_ROOT / "tests" / "data" / "regime_v1" / "block_features.npz"


def _banner(text: str) -> None:
    print(f"\n=== {text} ===")


def run_synthetic_pipeline(out: Path) -> dict:
    """Step 1: ingest -> validate -> analyse -> report on a synthetic case."""
    from metis.analysis import run_analysis
    from metis.data.registry import CaseRegistry
    from metis.reporting.generators import physics_report
    from metis.testing import mock_dns

    root = out / "synthetic_dns"
    if root.exists():
        shutil.rmtree(root)
    cid = "case_synthetic"
    _h5, meta = mock_dns.generate(root / "raw" / cid / "s_00001.h5", seed=0, case_id=cid)
    (root / "processed" / cid).mkdir(parents=True)
    shutil.move(str(meta), root / "processed" / cid / "metadata.json")

    reg = CaseRegistry(root)
    result = run_analysis(reg, "physics", cid)
    report_dir = physics_report(result).write(out / "reports" / "physics-synthetic")
    print(f"validated + analysed {cid} (synthetic, random fields): "
          f"validation ok={result.validation['ok']}")
    print(f"report: {report_dir / 'summary.md'}")
    return {"validation_ok": bool(result.validation["ok"]), "report": str(report_dir)}


def run_benchmark_on_sample(out: Path, sample: Path) -> tuple[bool, dict]:
    """Step 2: the frozen regime-v1 benchmark on the committed feature sample."""
    from metis.evaluation.benchmark import load_config, run_regime_v1
    from metis.features.regime import FEATURE_BLOCK_NAMES, CaseGridLabel
    from metis.reporting.generators import regime_v1_report

    with np.load(sample) as npz:
        blocks = {b: npz[b] for b in FEATURE_BLOCK_NAMES}
        labels = {
            str(c): CaseGridLabel(Pb_Pc=float(p), Thw_Tc=float(t))
            for c, p, t in zip(npz["case_ids"], npz["Pb_Pc"], npz["Thw_Tc"])
        }
    result = run_regime_v1(blocks, labels, load_config())
    print(result.summary())

    payload = {
        "name": result.name, "passed": result.passed,
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                   for c in result.checks],
        "blocks": result.blocks, "combined": result.combined,
        "provenance": {**result.provenance, "features": str(sample.name)},
    }
    (out / "regime_v1.json").write_text(json.dumps(payload, indent=2, default=str))
    report_dir = regime_v1_report(payload).write(out / "reports" / "regime-v1")
    print(f"report: {report_dir / 'summary.md'}")
    return result.passed, payload


def run_router_policy() -> list[dict]:
    """Step 3: score the curated scenarios with the OOD confidence rule."""
    from metis.router.confidence import confidence_score
    from metis.router.scenarios import SCENARIOS

    rows = []
    for s in SCENARIOS:
        conf = confidence_score({"Pb_Pc": s.Pb_Pc, "Thw_Tc": s.Thw_Tc, "Tcw_Tc": s.Tcw_Tc})
        source = "surrogate" if conf.trusts_surrogate() else "DNS database"
        rows.append({"scenario": s.name, "Pb_Pc": s.Pb_Pc, "Thw_Tc": s.Thw_Tc,
                     "Tcw_Tc": s.Tcw_Tc, "confidence": conf.level, "routed_to": source,
                     "reason": conf.reason})
        print(f"{s.name:14s} Tcw/Tc={s.Tcw_Tc:5.3f}  confidence={conf.level:6s} -> {source}")
    return rows


def summary_figure(benchmark: dict, routes: list[dict], path: Path) -> Path:
    """Two panels: which feature block recovers which axis (regime-v1), and
    where the router trusts the surrogate (Thw_Tc vs Tcw_Tc)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from metis.router.confidence import PSEUDOCRITICAL_RATIO, TRAINING_RANGES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    blocks = benchmark["blocks"]
    names = list(blocks)
    x = np.arange(len(names))
    ax1.bar(x - 0.2, [blocks[b]["loco_accuracy_Pb_Pc"] for b in names], 0.4,
            label="pressure ratio $P_b/P_c$", color="#3b6fb6")
    ax1.bar(x + 0.2, [blocks[b]["loco_accuracy_Thw_Tc"] for b in names], 0.4,
            label="hot-wall ratio $T_{hw}/T_c$", color="#d9822b")
    ax1.axhline(1 / 3, color="0.4", ls="--", lw=1)
    ax1.text(len(names) - 0.5, 1 / 3 + 0.02, "chance (3 classes)", ha="right",
             va="bottom", fontsize=8, color="0.3")
    ax1.set_xticks(x, [n.replace("_", " ") for n in names])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("leave-one-case-out accuracy")
    ax1.set_title("Regime discovery (9 DNS cases): each axis lives in a different block",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")

    lo, hi = TRAINING_RANGES["Tcw_Tc"]
    tlo, thi = TRAINING_RANGES["Thw_Tc"]
    ax2.axvspan(PSEUDOCRITICAL_RATIO, 1.1, color="#c0392b", alpha=0.08)
    ax2.add_patch(plt.Rectangle((lo, tlo), hi - lo, thi - tlo, fill=False,
                                ec="#2e8b57", fc="#2e8b57", alpha=0.6, lw=1.2))
    ax2.add_patch(plt.Rectangle((lo, tlo), hi - lo, thi - tlo, color="#2e8b57",
                                alpha=0.06, lw=0))
    ax2.text(lo + 0.003, thi + 0.008, "training envelope", color="#2e8b57", fontsize=8)
    ax2.axvline(PSEUDOCRITICAL_RATIO, color="#c0392b", lw=1.2)
    ax2.text(PSEUDOCRITICAL_RATIO + 0.004, 1.33, "pseudo-critical\n$T_{cw}/T_c=1$",
             color="#c0392b", fontsize=8, va="top")
    style = {"high": ("o", "#2e8b57"), "medium": ("s", "#d9822b"), "low": ("X", "#c0392b")}
    seen = set()
    for r in routes:
        m, c = style[r["confidence"]]
        lab = f"{r['confidence']} -> {r['routed_to']}"
        ax2.scatter(r["Tcw_Tc"], r["Thw_Tc"], marker=m, color=c, s=60, zorder=3,
                    label=None if lab in seen else lab)
        seen.add(lab)
        if not r["scenario"].startswith("case0"):
            ax2.annotate(r["scenario"], (r["Tcw_Tc"], r["Thw_Tc"]), fontsize=7,
                         xytext=(-10, 8), textcoords="offset points")
    ax2.text(0.875, 1.062, "training cases sit on the grid corners/centre\n"
             "(several share the same wall ratios)", ha="center", fontsize=7, color="0.35")
    ax2.set_xlim(0.75, 1.1)
    ax2.set_ylim(1.05, 1.45)
    ax2.set_xlabel("cold-wall ratio $T_{cw}/T_c$")
    ax2.set_ylabel("hot-wall ratio $T_{hw}/T_c$")
    ax2.set_title("Router: trust the surrogate only inside the validated envelope",
                  fontsize=10)
    ax2.legend(fontsize=7, loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(out: str | Path = "demo_output", sample: str | Path | None = None,
        figure: bool = True) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sample = Path(sample) if sample else DEFAULT_SAMPLE
    if not sample.exists():
        raise FileNotFoundError(
            f"feature sample not found at {sample}; run from a METIS checkout "
            "or pass --sample")

    _banner("1/3  Pipeline on a synthetic DNS case (random fields, not physics)")
    pipeline = run_synthetic_pipeline(out)

    _banner("2/3  Frozen regime-v1 benchmark on the committed real-DNS feature sample")
    passed, bench = run_benchmark_on_sample(out, sample)

    _banner("3/3  Router OOD confidence policy (decision only; no checkpoint needed)")
    routes = run_router_policy()
    (out / "router_decisions.json").write_text(json.dumps(routes, indent=2))

    fig_path = None
    if figure:
        try:
            fig_path = summary_figure(bench, routes, out / "summary.png")
        except ImportError:
            print("\n(matplotlib not installed: skipping summary.png; "
                  "pip install 'metis[report]')")

    _banner("Done")
    print(f"synthetic pipeline : validation ok = {pipeline['validation_ok']}")
    print(f"regime-v1 benchmark: {'PASS' if passed else 'FAIL'}")
    print(f"router scenarios   : {len(routes)} scored")
    if fig_path:
        print(f"summary figure     : {fig_path}")
    print(f"all outputs        : {out.resolve()}")
    return 0 if (passed and pipeline["validation_ok"]) else 1

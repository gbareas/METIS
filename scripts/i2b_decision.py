"""I2-B / B7 — the accept/stop decision gate.

Consumes ``results/i2b_evaluation.json`` (B6) and runs
`metis.evaluation.assessment.assess_model` (AE vs PCA) per seed at the
headline latent dim, applying the protocol §6 criteria:

  positive  — >=3/5 seeds `is_better=True` and the improvement exceeds the
              seed-to-seed std;
  negative  — no seed better, OR within seed noise, OR an ML metric
              improves while a physical diagnostic regresses, OR
              `latent_stability` is poor while PCA's latent is exact.

`is_better` (protocol §17) requires an ML metric to improve AND no
physical diagnostic to regress. Writes ``results/i2b_decision.json`` and
prints the verdict; FINDINGS §7 records it. No model is registered on a
negative.

    python scripts/i2b_decision.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from metis import tracking
from metis.evaluation.assessment import COMMON_LOWER_IS_BETTER, assess_model

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "results" / "i2b_evaluation.json"
OUT = REPO / "results" / "i2b_decision.json"

# protocol §5.5
_ML_METRICS = ("val_relative_l2", "ood_relative_l2", "ood_degradation_ratio")
_PHYS_NAMES = (
    "mean_profile_rel_l2_x", "mean_profile_rel_l2_z",
    "rms_profile_rel_l2_x", "rms_profile_rel_l2_z",
    "spectrum_rel_l2_x", "spectrum_rel_l2_z",
    "spectrum_log_corr_x", "spectrum_log_corr_z",
    "energy_fraction_l1",
)
_PHYS_METRICS = tuple(f"{s}_{n}" for s in ("val", "ood") for n in _PHYS_NAMES)
# a *lower* OOD-degradation ratio is better; assess_model can't infer that
_LOWER_IS_BETTER = (*COMMON_LOWER_IS_BETTER, "degradation_ratio")
_LATENT_STABILITY_TOL = 0.05   # PCA's latent is exact (0.0); AE above this is "poor"


def _flat(block: dict) -> dict:
    """`{val:{...}, ood:{...}, ood_degradation_ratio:x}` -> flat metric dict."""
    out = {"ood_degradation_ratio": block["ood_degradation_ratio"]}
    for split in ("val", "ood"):
        out[f"{split}_relative_l2"] = block[split]["relative_l2"]
        for n in _PHYS_NAMES:
            out[f"{split}_{n}"] = block[split][n]
    return out


def _assess_k(entry: dict) -> dict:
    pca_flat = _flat(entry["pca"])
    per_seed = []
    for row in entry["ae"]["per_seed"]:
        ae_flat = _flat(row)
        v = assess_model(
            ae_flat, pca_flat,
            ml_metrics=_ML_METRICS, physical_metrics=_PHYS_METRICS,
            lower_is_better=_LOWER_IS_BETTER,
        )
        per_seed.append({"seed": row["seed"], **asdict(v)})
    n_better = sum(p["is_better"] for p in per_seed)
    stab = entry["ae"]["latent_stability"]
    return {
        "n_seeds": len(per_seed),
        "n_is_better": n_better,
        "latent_stability_max_rel_l2": stab["max_rel_l2"],
        "latent_stability_poor": stab["max_rel_l2"] > _LATENT_STABILITY_TOL,
        "per_seed": per_seed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluation", default=EVAL, type=Path)
    p.add_argument("--output", default=OUT, type=Path)
    p.add_argument("--no-track", dest="track", action="store_false")
    args = p.parse_args(argv)

    ev = json.loads(args.evaluation.read_text())
    headline = str(ev["headline_latent_dim"])
    by_k = {k: _assess_k(entry) for k, entry in ev["per_latent_dim"].items()}
    head = by_k[headline]

    # protocol §6
    positive = head["n_is_better"] >= 3
    negative_reasons = []
    if head["n_is_better"] == 0:
        negative_reasons.append("no seed reaches assess_model.is_better=True")
    any_ml_up = any(pp["ml_improved"] for pp in head["per_seed"])
    any_phys_down = any(pp["physical_regressed"] for pp in head["per_seed"])
    if any_ml_up and any_phys_down:
        negative_reasons.append("an ML metric improves while a physical diagnostic regresses")
    if head["latent_stability_poor"]:
        negative_reasons.append(
            f"latent_stability is poor (AE Procrustes relL2 "
            f"{head['latent_stability_max_rel_l2']:.2f} across seeds; PCA is exact)"
        )

    decision = "POSITIVE — continue nonlinear representation learning" if positive else (
        "NEGATIVE — linear PCA/POD is sufficient for centre-plane u' structure "
        "across the tested thermodynamic conditions; STOP (no VAE / U-Net / operator escalation)"
    )

    payload = {
        "evaluation": str(args.evaluation.relative_to(REPO)),
        "headline_latent_dim": ev["headline_latent_dim"],
        "ml_metrics": list(_ML_METRICS),
        "physical_metrics": list(_PHYS_METRICS),
        "gate_rule": "is_better = an ML metric improves AND no physical diagnostic regresses",
        "headline": head,
        "by_latent_dim": {k: {"n_is_better": v["n_is_better"],
                              "latent_stability_max_rel_l2": v["latent_stability_max_rel_l2"]}
                          for k, v in by_k.items()},
        "positive": positive,
        "negative_reasons": negative_reasons,
        "decision": decision,
        "register_model": positive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"headline k={headline}: {head['n_is_better']}/{head['n_seeds']} seeds is_better=True")
    for pp in head["per_seed"]:
        print(f"  seed {pp['seed']}: is_better={pp['is_better']!s:5}  {pp['reason']}")
    nseeds = len(ev["seeds"])
    print("\nis_better by k: " + "  ".join(
        f"k{k}={v['n_is_better']}/{nseeds}" for k, v in by_k.items()))
    print("latent_stability max relL2 by k: " + "  ".join(
        f"k{k}={v['latent_stability_max_rel_l2']:.2f}" for k, v in by_k.items()))
    if negative_reasons:
        print("\nnegative criteria met:")
        for r in negative_reasons:
            print(f"  - {r}")
    print(f"\nDECISION: {decision}")
    print(f"Wrote {args.output}")

    with tracking.run("i2b-representation", run_name="decision",
                      params={"headline_latent_dim": ev["headline_latent_dim"],
                              "ml_metrics": list(_ML_METRICS)},
                      tags={"phase": "B7"}, enabled=args.track) as run:
        run.log_metrics({
            "n_is_better_headline": head["n_is_better"],
            "latent_stability_max_rel_l2_headline": head["latent_stability_max_rel_l2"],
            "positive": float(positive),
        })
        run.log_dict(payload, "i2b_decision.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report generators for the artifacts METIS already produces (I6).

Each takes a parsed result (a dict from `results/*.json`, or an
`AnalysisResult`) and returns a `Report`; the CLI or a caller then
`.write(out_dir)`s it.
"""
from __future__ import annotations

import numpy as np

from metis.reporting.report import Report

_BLOCK_METRICS = ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc", "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc")


# --------------------------------------------------------------------- #
# regime-v1 benchmark  (results/regime_v1.json)
# --------------------------------------------------------------------- #
def regime_v1_report(data: dict) -> Report:
    r = Report(title=f"Benchmark: {data.get('name', 'regime-v1')}")
    r.provenance.update(data.get("provenance", {}))

    passed = data["passed"]
    checks = data["checks"]
    r.section(
        "Result",
        f"**{'PASS' if passed else 'FAIL'}** — "
        f"{sum(c['passed'] for c in checks)}/{len(checks)} frozen checks passed.",
    )
    r.section("Checks", table=[
        {"check": c["name"], "passed": c["passed"], "detail": c["detail"]} for c in checks
    ])

    blocks = data["blocks"]
    r.section("Per-block metrics", table=[
        {"block": b, **{m: blocks[b][m] for m in _BLOCK_METRICS}} for b in blocks
    ])
    combined = data["combined"]
    r.section("Combined (MFA vs naive) — OOD nearest Pb_Pc centroid", table=[
        {"method": m, **{k: v for k, v in combined[m]["ood_nearest_Pb_Pc_centroid"].items()}}
        for m in combined
    ])

    for b in blocks:
        for m in _BLOCK_METRICS:
            r.metrics[f"{b}.{m}"] = blocks[b][m]
    r.metrics["passed"] = float(passed)

    def _fig(fig):
        ax = fig.subplots()
        names = list(blocks)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [blocks[b]["ari_vs_Pb_Pc"] for b in names], 0.4, label="ARI vs Pb_Pc")
        ax.bar(x + 0.2, [blocks[b]["ari_vs_Thw_Tc"] for b in names], 0.4, label="ARI vs Thw_Tc")
        ax.set_xticks(x, names)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("adjusted Rand index")
        ax.set_title("regime-v1: per-block cluster agreement")
        ax.legend()

    r.add_figure("block_ari", _fig)
    return r


# --------------------------------------------------------------------- #
# representation study  (results/representation_study.json)
# --------------------------------------------------------------------- #
def representation_study_report(data: dict) -> Report:
    r = Report(title="Representation study (I2-A)")
    r.section("Verdict", data["verdict"])

    blocks = data["blocks"]
    rows = []
    for b, res in blocks.items():
        ae = res["autoencoder"]
        rows.append({
            "block": b,
            "n_feat": res["n_features"],
            "PCA ARI Pb/Thw": f"{res['pca']['ari_vs_Pb_Pc']:+.3f} / {res['pca']['ari_vs_Thw_Tc']:+.3f}",
            "AE ARI Pb/Thw": f"{ae['ari_vs_Pb_Pc_mean']:+.3f} / {ae['ari_vs_Thw_Tc_mean']:+.3f}",
            "AE latent stability": ae.get("latent_stability", {}).get("mean_rel_l2"),
            "AE beats PCA": res["autoencoder_vs_pca"]["beats_baseline"],
        })
    r.section("PCA vs Autoencoder (latent dim 2)", table=rows)

    for b, res in blocks.items():
        r.metrics[f"{b}.pca.ari_vs_Pb_Pc"] = res["pca"]["ari_vs_Pb_Pc"]
        r.metrics[f"{b}.pca.ari_vs_Thw_Tc"] = res["pca"]["ari_vs_Thw_Tc"]
        r.metrics[f"{b}.ae.ari_vs_Pb_Pc_mean"] = res["autoencoder"]["ari_vs_Pb_Pc_mean"]
        r.metrics[f"{b}.ae.ari_vs_Thw_Tc_mean"] = res["autoencoder"]["ari_vs_Thw_Tc_mean"]
        r.metrics[f"{b}.ae_beats_pca"] = float(res["autoencoder_vs_pca"]["beats_baseline"])
    r.metrics["any_block_ae_beats_pca"] = float(data["any_block_ae_beats_pca"])

    def _fig(fig):
        axes = fig.subplots(1, 2)
        for ax, axis_key, ttl in zip(axes, ("Pb_Pc", "Thw_Tc"), ("pressure", "thermal")):
            names = list(blocks)
            x = np.arange(len(names))
            ax.bar(x - 0.2, [blocks[b]["pca"][f"ari_vs_{axis_key}"] for b in names], 0.4, label="PCA")
            ax.bar(x + 0.2, [blocks[b]["autoencoder"][f"ari_vs_{axis_key}_mean"] for b in names],
                   0.4, label="AE")
            ax.set_xticks(x, names)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"ARI vs {axis_key} ({ttl})")
            ax.legend()

    r.add_figure("pca_vs_ae_ari", _fig)
    return r


# --------------------------------------------------------------------- #
# case physics summary  (an AnalysisResult from run_analysis(..., "physics"))
# --------------------------------------------------------------------- #
def physics_report(result) -> Report:
    case = result.case_ids[0]
    r = Report(title=f"Physics summary: {case}")
    r.provenance.update(result.provenance)

    out = result.outputs
    bulk_keys = ("rho_b", "U_b", "T_b", "P_b", "Re_b", "Pr_b", "Ec_b", "Br_b", "Ma_b")
    wall_keys = ("T_cw", "u_tau_cw", "Re_tau_cw", "T_hw", "u_tau_hw", "Re_tau_hw")
    r.section("Bulk state & dimensionless groups",
              table=[{"quantity": k, "value": out[k]} for k in bulk_keys if k in out])
    r.section("Wall friction Reynolds",
              table=[{"quantity": k, "value": out[k]} for k in wall_keys if k in out])
    r.add_metrics({k: out[k] for k in (*bulk_keys, *wall_keys) if k in out})

    if result.validation is not None:
        r.section("Validation",
                  f"{'OK' if result.validation['ok'] else 'FAILED'} — "
                  f"{result.validation['errors']} error(s), {result.validation['warnings']} warning(s).")

    arrays = result.arrays
    if "y" in arrays and "avg_T" in arrays:
        def _fig(fig):
            axes = fig.subplots(1, 2)
            y = np.asarray(arrays["y"])
            for ax, fld in zip(axes, ("avg_u", "avg_T")):
                if fld in arrays:
                    ax.plot(np.asarray(arrays[fld]), y)
                    ax.set_xlabel(fld)
                    ax.set_ylabel("y [m]")
            fig.suptitle(f"{case}: wall-normal mean profiles")

        r.add_figure("mean_profiles", _fig)
    return r


GENERATORS = {
    "regime-v1": regime_v1_report,
    "representation": representation_study_report,
}

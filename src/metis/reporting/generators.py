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
# I2-B slice representation study  (a bundle of results/i2b_*.json)
# --------------------------------------------------------------------- #
_I2B_PHYS_ROWS = (
    ("rms_profile_rel_l2_x", "RMS profile relL2 (x)"),
    ("rms_profile_rel_l2_z", "RMS profile relL2 (z)"),
    ("spectrum_rel_l2_x", "spectrum relL2 (x)"),
    ("spectrum_rel_l2_z", "spectrum relL2 (z)"),
    ("spectrum_log_corr_x", "log-spectrum corr (x)"),
    ("spectrum_log_corr_z", "log-spectrum corr (z)"),
    ("energy_fraction_l1", "POD energy-fraction L1"),
)


def i2b_representation_report(data: dict) -> Report:
    """`data` bundles the study's result JSONs by key: `evaluation` and
    `decision` are required; `training`, `baseline`, `modal_compare` are
    used when present. Built by `metis report i2b-representation --from`."""
    ev = data["evaluation"]
    dec = data["decision"]
    train = data.get("training", {})
    modal = data.get("modal_compare")
    hk = str(ev["headline_latent_dim"])
    grid = [str(k) for k in ev["latent_dims"]]

    r = Report(title="I2-B slice-level representation study")
    r.provenance.update({"headline_latent_dim": ev["headline_latent_dim"],
                         "seeds": ev.get("seeds"), "dataset": ev.get("dataset")})

    r.section("Verdict", f"**{dec['decision']}**\n\n"
              + "\n".join(f"- {why}" for why in dec.get("negative_reasons", []))
              + f"\n\nModel registered: **{'yes' if dec.get('register_model') else 'no'}**.")

    if train:
        r.section("Setup", table=[{
            "field": train.get("field"), "slice": train.get("slice_id"),
            "grid": "x".join(map(str, train.get("input_hw", []))),
            "conv channels": ",".join(map(str, train.get("channels", []))),
            "activation": train.get("activation"), "batch": train.get("batch_size"),
            "latent dims": ",".join(grid), "seeds": len(ev.get("seeds", [])),
            "n train/val": f"{train.get('n', {}).get('train')}/{train.get('n', {}).get('val')}",
        }])

    recon_rows = []
    for k in grid:
        e = ev["per_latent_dim"][k]
        recon_rows.append({
            "k": k, "compression": round(e["compression_ratio"]),
            "PCA val relL2": e["pca"]["val"]["relative_l2"],
            "AE val relL2": e["ae"]["val_mean"]["relative_l2"],
            "AE val std": e["ae"]["val_std"]["relative_l2"],
            "PCA OOD relL2": e["pca"]["ood"]["relative_l2"],
            "AE OOD relL2": e["ae"]["ood_mean"]["relative_l2"],
            "PCA OOD-degr": e["pca"]["ood_degradation_ratio"],
            "AE OOD-degr": e["ae"]["ood_degradation_ratio_mean"],
            "AE latent-stab max": e["ae"]["latent_stability"]["max_rel_l2"],
        })
    r.section("Reconstruction & robustness vs latent dim "
              "(PCA is the closed-form optimum; lower relL2 better)", table=recon_rows)

    he = ev["per_latent_dim"][hk]
    phys_rows = []
    for key, label in _I2B_PHYS_ROWS:
        phys_rows.append({
            "diagnostic": label,
            "PCA val": he["pca"]["val"][key], "AE val": he["ae"]["val_mean"][key],
            "PCA OOD": he["pca"]["ood"][key], "AE OOD": he["ae"]["ood_mean"][key],
        })
    r.section(f"Physical fidelity at headline k={hk} (true vs reconstruction)", table=phys_rows)

    r.section(f"assess_model per seed at k={hk} "
              "(is_better needs an ML gain AND no physical regression)",
              table=[{"seed": p["seed"], "is_better": p["is_better"], "reason": p["reason"]}
                     for p in dec["headline"]["per_seed"]])

    if modal:
        lr = modal["summary"]["linear_rank"]
        ef = modal["splits"]["val"]["pod_energy_fraction"]
        r.section("Modal structure",
                  f"Training POD is high-rank: **{lr['modes_for_90pct']} modes for 90 %**, "
                  f"**{lr['modes_for_99pct']} for 99 %**; k={hk} captures "
                  f"{lr['reconstructed_variance_at_k'] * 100:.1f} %. The conv-AE spans "
                  "essentially the same leading subspace as POD — reconstructed-ensemble "
                  "POD energy fractions (val):",
                  table=[{"mode": i + 1, "true": round(ef["true"][i], 3),
                          "PCA recon": round(ef["pca"][i], 3), "AE recon": round(ef["ae"][i], 3)}
                         for i in range(min(6, len(ef["true"])))])

    r.metrics.update({
        f"k{hk}.pca_val_relL2": he["pca"]["val"]["relative_l2"],
        f"k{hk}.ae_val_relL2": he["ae"]["val_mean"]["relative_l2"],
        f"k{hk}.pca_ood_relL2": he["pca"]["ood"]["relative_l2"],
        f"k{hk}.ae_ood_relL2": he["ae"]["ood_mean"]["relative_l2"],
        f"k{hk}.ae_latent_stability_max": he["ae"]["latent_stability"]["max_rel_l2"],
        f"k{hk}.n_is_better": dec["headline"]["n_is_better"],
        "positive": float(bool(dec.get("positive"))),
        "register_model": float(bool(dec.get("register_model"))),
    })
    if modal:
        r.metrics["modes_for_90pct"] = modal["summary"]["linear_rank"]["modes_for_90pct"]
        r.metrics["modes_for_99pct"] = modal["summary"]["linear_rank"]["modes_for_99pct"]

    def _recon_fig(fig):
        ax = fig.subplots()
        ks = [int(k) for k in grid]
        for split, style in (("val", "-"), ("ood", "--")):
            ax.plot(ks, [ev["per_latent_dim"][k]["pca"][split]["relative_l2"] for k in grid],
                    style, marker="o", label=f"PCA {split}")
            ax.plot(ks, [ev["per_latent_dim"][k]["ae"][f"{split}_mean"]["relative_l2"] for k in grid],
                    style, marker="s", label=f"conv-AE {split}")
        ax.set(xlabel="latent dim k", ylabel="reconstruction relative L2",
               title="I2-B: conv-AE never beats PCA")
        ax.legend()

    r.add_figure("reconstruction_vs_k", _recon_fig)

    if modal and modal.get("training_cumulative_energy_head"):
        cum = modal["training_cumulative_energy_head"]

        def _pod_fig(fig):
            ax = fig.subplots()
            ax.plot(range(1, len(cum) + 1), cum, marker=".")
            ax.axhline(0.9, ls="--", c="grey")
            ax.axhline(0.99, ls=":", c="grey")
            ax.set(xlabel="POD mode", ylabel="cumulative reconstructed variance",
                   title="Training POD spectrum (centre-plane u')")

        r.add_figure("pod_cumulative_spectrum", _pod_fig)

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
    "i2b-representation": i2b_representation_report,
}

"""Tests for metis.reporting (I6)."""
import json

import pytest

from metis.reporting import Report, has_matplotlib
from metis.reporting.generators import (
    i2b_representation_report,
    regime_v1_report,
    representation_study_report,
)

# --- minimal sample results (shapes match results/*.json) -----
REGIME_V1 = {
    "name": "regime-v1",
    "passed": True,
    "checks": [
        {"name": "best_block__ari_vs_Pb_Pc", "passed": True, "detail": "bulk"},
        {"name": "no_metric_drift", "passed": True, "detail": "ok"},
    ],
    "blocks": {
        "bulk": {"ari_vs_Pb_Pc": 0.357, "ari_vs_Thw_Tc": 0.071,
                 "loco_accuracy_Pb_Pc": 0.556, "loco_accuracy_Thw_Tc": 0.111},
        "rms_profile": {"ari_vs_Pb_Pc": -0.161, "ari_vs_Thw_Tc": 0.484,
                        "loco_accuracy_Pb_Pc": 0.222, "loco_accuracy_Thw_Tc": 0.778},
    },
    "combined": {
        "mfa": {"ood_nearest_Pb_Pc_centroid": {"case10": 1.5, "case15": 1.5}},
        "naive": {"ood_nearest_Pb_Pc_centroid": {"case10": 5.0, "case15": 5.0}},
    },
    "provenance": {"code_version": "abc123"},
}

REPRESENTATION = {
    "verdict": "STOP CRITERION MET",
    "any_block_ae_beats_pca": False,
    "blocks": {
        "bulk": {
            "n_features": 7,
            "pca": {"ari_vs_Pb_Pc": 0.357, "ari_vs_Thw_Tc": 0.071},
            "autoencoder": {"ari_vs_Pb_Pc_mean": 0.353, "ari_vs_Thw_Tc_mean": -0.118,
                            "latent_stability": {"mean_rel_l2": 1e-16}},
            "autoencoder_vs_pca": {"beats_baseline": False},
        },
    },
}


# --- Report builder ---------------------------------------------
def test_report_writes_summary_and_metrics(tmp_path):
    r = (Report("T")
         .section("Intro", "hello", table=[{"a": 1, "b": 2.5}, {"a": 3, "b": 4.0}])
         .add_metrics({"x": 1.0, "y": 2}))
    out = r.write(tmp_path / "rep")
    md = (out / "summary.md").read_text()
    assert "# T" in md and "## Intro" in md and "| a | b |" in md and "2.5" in md
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["metrics"] == {"x": 1.0, "y": 2}
    assert "git_commit" in metrics["provenance"]


def test_figure_is_written_or_gracefully_skipped(tmp_path):
    r = Report("F").add_figure("plot", lambda fig: fig.subplots().plot([0, 1], [0, 1]))
    out = r.write(tmp_path / "f")
    md = (out / "summary.md").read_text()
    if has_matplotlib():
        assert (out / "figures" / "plot.png").exists()
        assert "![plot](figures/plot.png)" in md
    else:
        assert "skipped (matplotlib not installed" in md


# --- generators ------------------------------------------------
def test_regime_v1_report(tmp_path):
    out = regime_v1_report(REGIME_V1).write(tmp_path / "r")
    md = (out / "summary.md").read_text()
    assert "**PASS** — 2/2" in md
    assert "best_block__ari_vs_Pb_Pc" in md          # checks table
    assert "| block |" in md                          # per-block table
    m = json.loads((out / "metrics.json").read_text())["metrics"]
    assert m["bulk.ari_vs_Pb_Pc"] == 0.357 and m["passed"] == 1.0


def test_representation_study_report(tmp_path):
    out = representation_study_report(REPRESENTATION).write(tmp_path / "r")
    md = (out / "summary.md").read_text()
    assert "STOP CRITERION MET" in md
    assert "PCA vs Autoencoder" in md
    m = json.loads((out / "metrics.json").read_text())["metrics"]
    assert m["any_block_ae_beats_pca"] == 0.0
    assert m["bulk.ae_beats_pca"] == 0.0


@pytest.mark.skipif(not has_matplotlib(), reason="report extra (matplotlib) not installed")
def test_generators_produce_figures(tmp_path):
    out = regime_v1_report(REGIME_V1).write(tmp_path / "r")
    assert (out / "figures" / "block_ari.png").exists()


# --- I2-B representation study (a bundle of results/i2b_*.json) ---
_PHYS = {
    "rms_profile_rel_l2_x": 0.24, "rms_profile_rel_l2_z": 0.24,
    "spectrum_rel_l2_x": 0.35, "spectrum_rel_l2_z": 0.32,
    "spectrum_log_corr_x": 0.8, "spectrum_log_corr_z": 0.7,
    "energy_fraction_l1": 0.21,
}


def _k_entry(pca_val, ae_val):
    block = lambda rl2: {"relative_l2": rl2, **_PHYS}
    return {
        "compression_ratio": 288.0,
        "pca": {"val": block(pca_val), "ood": block(pca_val - 0.3),
                "ood_degradation_ratio": 0.54},
        "ae": {"per_seed": [], "val_mean": block(ae_val), "val_std": block(0.01),
               "ood_mean": block(ae_val - 0.23), "ood_std": block(0.01),
               "ood_degradation_ratio_mean": 0.66, "ood_degradation_ratio_std": 0.02,
               "latent_stability": {"mean_rel_l2": 0.25, "max_rel_l2": 0.38, "n": 5}},
    }


I2B_BUNDLE = {
    "evaluation": {
        "headline_latent_dim": 32, "latent_dims": [16, 32], "seeds": [0, 1, 2, 3, 4],
        "dataset": "artifacts/datasets/i2b_representation_v1_primary_u_s3_center",
        "per_latent_dim": {"16": _k_entry(0.73, 0.77), "32": _k_entry(0.65, 0.69)},
    },
    "decision": {
        "headline_latent_dim": 32,
        "decision": "NEGATIVE — linear PCA/POD is sufficient; STOP",
        "negative_reasons": ["no seed reaches assess_model.is_better=True"],
        "positive": False, "register_model": False,
        "headline": {"n_is_better": 0, "per_seed": [
            {"seed": s, "is_better": False, "reason": "no ML metric improved"} for s in range(5)
        ]},
    },
    "training": {
        "field": "u", "slice_id": "s3_center", "input_hw": [96, 96],
        "channels": [16, 32, 64], "activation": "gelu", "batch_size": 64,
        "n": {"train": 3600, "val": 900},
    },
    "modal_compare": {
        "training_cumulative_energy_head": [0.2, 0.5, 0.8, 0.9, 0.95, 0.99],
        "summary": {"linear_rank": {"reconstructed_variance_at_k": 0.776,
                                    "modes_for_90pct": 67, "modes_for_99pct": 181}},
        "splits": {"val": {"pod_energy_fraction": {
            "true": [0.16, 0.12, 0.09], "pca": [0.26, 0.18, 0.12], "ae": [0.26, 0.18, 0.10],
        }}},
    },
}


def test_i2b_representation_report(tmp_path):
    out = i2b_representation_report(I2B_BUNDLE).write(tmp_path / "r")
    md = (out / "summary.md").read_text()
    assert "NEGATIVE" in md and "Model registered: **no**" in md
    assert "67 modes for 90 %" in md
    assert "| k | compression |" in md
    assert "assess_model per seed" in md
    m = json.loads((out / "metrics.json").read_text())["metrics"]
    assert m["k32.n_is_better"] == 0
    assert m["register_model"] == 0.0
    assert m["modes_for_90pct"] == 67


def test_i2b_representation_report_without_optional_bundles(tmp_path):
    minimal = {k: I2B_BUNDLE[k] for k in ("evaluation", "decision")}
    out = i2b_representation_report(minimal).write(tmp_path / "r")
    md = (out / "summary.md").read_text()
    assert "NEGATIVE" in md
    assert "Modal structure" not in md          # modal_compare section skipped
    assert "reconstruction_vs_k" in md          # the eval-only figure still renders

"""Tests for metis.reporting (I6)."""
import json

import pytest

from metis.reporting import Report, has_matplotlib
from metis.reporting.generators import regime_v1_report, representation_study_report

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

"""Tests for metis.analysis — the standard analysis API (R6)."""
import shutil

import numpy as np
import pytest

from metis.analysis import ANALYSES, AnalysisResult, run_analysis
from metis.data.registry import CaseRegistry
from metis.features.regime import FEATURE_NAMES, RICH_POD_SLICE_IDS
from metis.testing import mock_dns, mock_slices

SLICE_ID = "s3_center"


def _make_case(root, case_id, seed):
    raw = root / "raw" / case_id
    _h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=6, ny=10, nz=6, seed=seed,
                                  case_id=case_id)
    proc = root / "processed" / case_id
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    for sid in RICH_POD_SLICE_IDS:
        mock_slices.generate(root / "processed_slices" / case_id / sid,
                             nx=8, nz=8, n_snapshots=16, seed=seed,
                             case_id=case_id, slice_id=sid)


@pytest.fixture
def registry(tmp_path):
    _make_case(tmp_path, "case01", seed=1)
    return CaseRegistry(tmp_path)


def test_all_four_analyses_are_registered():
    assert set(ANALYSES) == {"physics", "spectra", "pod", "regime_features"}


def test_physics_analysis(registry):
    r = run_analysis(registry, "physics", "case01")
    assert r.analysis == "physics" and r.case_ids == ["case01"]
    assert {"Re_b", "Pr_b", "Re_tau_cw", "Re_tau_hw"} <= set(r.outputs)
    assert "y" in r.arrays and "avg_T" in r.arrays
    assert r.validation is not None and r.validation["ok"]


def test_spectra_analysis_parseval(registry):
    r = run_analysis(registry, "spectra", "case01", slice_id=SLICE_ID, field="u", axis="x")
    assert r.config == {"slice_id": SLICE_ID, "field": "u", "axis": "x"}
    assert set(r.arrays) == {"k", "E", "kE"}
    assert r.outputs["parseval_rel_error"] < 1e-6


def test_pod_analysis(registry):
    r = run_analysis(registry, "pod", "case01", slice_id=SLICE_ID, field="u")
    assert r.outputs["rank"] >= 1
    assert 0 < r.outputs["energy_captured"] <= 1.0 + 1e-9
    assert r.arrays["leading_mode"].shape == (8, 8)
    assert r.arrays["energy_fractions"].ndim == 1


def test_regime_features_analysis(registry):
    r = run_analysis(registry, "regime_features", "case01", feature_set="compact")
    assert r.outputs["n_features"] == len(FEATURE_NAMES)
    assert r.outputs["feature_names"] == list(FEATURE_NAMES)
    assert r.arrays["features"].shape == (len(FEATURE_NAMES),)


def test_validate_false_skips_validation(registry):
    r = run_analysis(registry, "physics", "case01", validate=False)
    assert r.validation is None


def test_unknown_analysis_raises_listing_choices(registry):
    with pytest.raises(ValueError, match="physics"):
        run_analysis(registry, "not_an_analysis", "case01")


def test_unknown_case_raises_keyerror(registry):
    with pytest.raises(KeyError):
        run_analysis(registry, "physics", "case99")


def test_result_save_load_round_trip(registry, tmp_path):
    r = run_analysis(registry, "pod", "case01", slice_id=SLICE_ID)
    out = tmp_path / "res"
    r.save(out)
    back = AnalysisResult.load(out)
    assert back.analysis == r.analysis
    assert back.outputs == r.outputs
    assert back.config == r.config
    for key, arr in r.arrays.items():
        np.testing.assert_array_equal(back.arrays[key], arr)


def test_summary_mentions_analysis_and_case(registry):
    s = run_analysis(registry, "physics", "case01").summary()
    assert "physics" in s and "case01" in s


def test_provenance_is_populated(registry):
    r = run_analysis(registry, "physics", "case01")
    assert r.provenance["data_root"] == str(registry.data_root)
    assert "created_at" in r.provenance and "code_version" in r.provenance

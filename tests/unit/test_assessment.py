"""Tests for metis.evaluation.assessment — the §17 "is it better?" gate (I5)."""
from metis.evaluation.assessment import assess_model, is_lower_better

ML = ("rmse", "r2")
PHYS = ("mean_profile_rel_l2", "spectrum_log_corr")


def _v(candidate, baseline):
    return assess_model(candidate, baseline, ml_metrics=ML, physical_metrics=PHYS)


def test_lower_is_better_classification():
    assert is_lower_better("rmse") and is_lower_better("mean_profile_rel_l2")
    assert not is_lower_better("r2") and not is_lower_better("spectrum_log_corr")


def test_better_when_ml_improves_and_physics_holds():
    base = {"rmse": 1.0, "r2": 0.5, "mean_profile_rel_l2": 0.2, "spectrum_log_corr": 0.9}
    cand = {**base, "rmse": 0.8}                       # lower rmse = better
    v = _v(cand, base)
    assert v.is_better and v.ml_improved == ["rmse"] and not v.physical_regressed


def test_not_better_when_a_physical_diagnostic_regresses():
    base = {"rmse": 1.0, "r2": 0.5, "mean_profile_rel_l2": 0.2, "spectrum_log_corr": 0.9}
    cand = {**base, "rmse": 0.5, "mean_profile_rel_l2": 0.6}   # ML up, profile worse
    v = _v(cand, base)
    assert not v.is_better
    assert v.ml_improved == ["rmse"]
    assert v.physical_regressed == ["mean_profile_rel_l2"]
    assert "regressed" in v.reason


def test_not_better_when_no_ml_metric_improves():
    base = {"rmse": 1.0, "r2": 0.5, "mean_profile_rel_l2": 0.2, "spectrum_log_corr": 0.9}
    cand = {**base, "mean_profile_rel_l2": 0.05, "spectrum_log_corr": 0.99}  # physics up, ML flat
    v = _v(cand, base)
    assert not v.is_better and v.reason == "no ML metric improved"
    assert set(v.physical_improved) == {"mean_profile_rel_l2", "spectrum_log_corr"}


def test_deltas_are_signed_so_positive_means_better():
    base = {"rmse": 1.0, "r2": 0.5, "mean_profile_rel_l2": 0.2, "spectrum_log_corr": 0.9}
    cand = {"rmse": 0.7, "r2": 0.6, "mean_profile_rel_l2": 0.1, "spectrum_log_corr": 0.95}
    v = _v(cand, base)
    assert v.ml_deltas["rmse"] > 0            # rmse went down -> positive delta
    assert v.physical_deltas["mean_profile_rel_l2"] > 0

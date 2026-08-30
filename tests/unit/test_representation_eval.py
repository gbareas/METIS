"""Tests for metis.evaluation.representation (I2)."""
import numpy as np

from metis.evaluation.representation import compare_to_baseline, evaluate_representation

# 9 training cases on a 3x3 (Pb_Pc, Thw_Tc) grid, like the real benchmark.
PB = np.repeat([1.5, 2.0, 5.0], 3)
THW = np.tile([1.1, 1.2, 1.4], 3)


def test_latent_aligned_with_pressure_scores_high_ari_pb():
    # a 1-D latent that IS the pressure level -> perfect pressure clustering
    Z = np.c_[PB + 1e-3 * np.arange(9), np.zeros(9)]
    r = evaluate_representation(Z, PB, THW)
    assert r["ari_vs_Pb_Pc"] == 1.0
    assert r["ari_vs_Thw_Tc"] < 0.5


def test_ood_nearest_centroid_uses_pressure_geometry():
    Z = np.c_[PB, np.zeros(9)]
    Z_ood = np.array([[1.55, 0.0], [4.8, 0.0]])
    r = evaluate_representation(Z, PB, THW, Z_ood=Z_ood, ood_case_ids=["case10", "case15"])
    assert r["ood_nearest_Pb_Pc_centroid"] == {"case10": 1.5, "case15": 5.0}


def test_compare_to_baseline_flags_improvement_and_regression():
    base = {"ari_vs_Pb_Pc": 0.3, "ari_vs_Thw_Tc": 0.4,
            "loco_accuracy_Pb_Pc": 0.5, "loco_accuracy_Thw_Tc": 0.5}
    better = {**base, "ari_vs_Pb_Pc": 0.6}
    mixed = {**base, "ari_vs_Pb_Pc": 0.6, "ari_vs_Thw_Tc": 0.1}

    assert compare_to_baseline(better, base)["beats_baseline"] is True
    assert compare_to_baseline(better, base)["improved"] == ["ari_vs_Pb_Pc"]
    assert compare_to_baseline(mixed, base)["beats_baseline"] is False
    assert compare_to_baseline(base, base)["beats_baseline"] is False


def test_seed_changes_are_reported_via_evaluate_call():
    Z = np.random.default_rng(0).normal(size=(9, 2))
    a = evaluate_representation(Z, PB, THW, seed=0)
    b = evaluate_representation(Z, PB, THW, seed=1)
    assert set(a) == set(b)  # same keys regardless of seed

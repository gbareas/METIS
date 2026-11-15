"""Tests for metis.evaluation.representation (I2)."""
import numpy as np
import pytest

from metis.evaluation.representation import (
    compare_to_baseline,
    evaluate_representation,
    latent_physical_correlation,
    latent_stability,
)

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


# --- I5 additions: latent stability + physical correlation ---
def test_latent_stability_zero_for_rotations_of_the_same_embedding():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(12, 3))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0, 0, -1]])                       # rotation + reflection
    s = latent_stability([Z, Z @ rot, -Z])
    assert s["mean_rel_l2"] == pytest.approx(0.0, abs=1e-9)
    assert s["n"] == 3


def test_latent_stability_grows_with_added_noise():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(12, 2))
    quiet = latent_stability([Z, Z + 0.01 * rng.normal(size=Z.shape)])
    noisy = latent_stability([Z, Z + 0.5 * rng.normal(size=Z.shape)])
    assert noisy["mean_rel_l2"] > quiet["mean_rel_l2"]


def test_latent_physical_correlation_finds_the_encoding_axis():
    v = np.linspace(0, 1, 20)
    Z = np.c_[np.random.default_rng(0).normal(size=20), 3 * v + 0.001]
    cor = latent_physical_correlation(Z, {"pressure": v})
    assert cor["pressure"] == pytest.approx(1.0, abs=1e-3)


# --- I2-B B3: latent-dim selection from the PCA spectrum ---
def test_explained_variance_curve_is_cumulative_and_ends_at_one():
    from metis.evaluation.representation import explained_variance_curve
    cum = explained_variance_curve([3.0, 2.0, 1.0])
    assert cum[-1] == pytest.approx(1.0)
    assert np.all(np.diff(cum) >= 0)
    assert cum[0] == pytest.approx(9 / 14)


def test_choose_latent_dim_picks_smallest_k_over_threshold():
    from metis.evaluation.representation import choose_latent_dim
    # spectrum where k=3 first crosses 0.9
    sv = np.sqrt([50, 30, 15, 4, 1])
    c = choose_latent_dim(sv, [2, 3, 4, 8], threshold=0.9)
    assert c["headline_latent_dim"] == 3 and c["reached"] is True
    assert c["reconstructed_variance"] >= 0.9


def test_choose_latent_dim_falls_back_to_largest_when_unreachable():
    from metis.evaluation.representation import choose_latent_dim
    sv = np.sqrt([10, 9, 8, 7, 6, 5])   # very flat -> no small k hits 0.9
    c = choose_latent_dim(sv, [2, 4], threshold=0.9)
    assert c["headline_latent_dim"] == 4 and c["reached"] is False
    assert c["reconstructed_variance"] < 0.9

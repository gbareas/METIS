import numpy as np
import pytest

from metis.evaluation.regime import (
    adjusted_rand_index,
    combine_blocks_mfa,
    evaluate_regime_discovery,
    kmeans_cluster,
    loco_axis_accuracy,
    mfa_normalize_block,
    pca,
    standardize,
)
from metis.features.regime import ALL_CASE_IDS, TRAIN_CASE_IDS, CaseGridLabel


def test_standardize_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[10, -5, 0], scale=[2, 0.5, 1], size=(50, 3))
    Xz = standardize(X)
    np.testing.assert_allclose(Xz.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Xz.std(axis=0), 1.0, atol=1e-10)


def test_standardize_handles_zero_variance_column():
    X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
    Xz = standardize(X)
    np.testing.assert_allclose(Xz[:, 0], 0.0)


def test_pca_reconstructs_centered_data_at_full_rank():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 5))
    scores, explained, components = pca(X)
    reconstructed = scores @ components
    np.testing.assert_allclose(reconstructed, X - X.mean(axis=0), atol=1e-8)
    assert explained.sum() == pytest.approx(1.0, rel=1e-10)


def test_pca_rank_one_data_has_one_dominant_component():
    direction = np.array([1.0, 2.0, -1.0])
    rng = np.random.default_rng(2)
    coeffs = rng.normal(size=(20, 1))
    X = coeffs @ direction[None, :]
    _scores, explained, _components = pca(X)
    assert explained[0] == pytest.approx(1.0, abs=1e-8)


def test_adjusted_rand_index_identical_labelings_is_one():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_adjusted_rand_index_invariant_to_relabeling():
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([5, 5, 5, 3, 3, 3, 7, 7, 7])
    assert adjusted_rand_index(labels_true, labels_pred) == pytest.approx(1.0)


def test_adjusted_rand_index_matches_closed_form_worst_case():
    # Every true group is split evenly across every predicted group -> a
    # known closed-form ARI of -1/3 for 3 groups of 3.
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert adjusted_rand_index(labels_true, labels_pred) == pytest.approx(-1 / 3)


def test_kmeans_recovers_well_separated_clusters():
    rng = np.random.default_rng(3)
    centers = np.array([[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]])
    X = np.concatenate([c + rng.normal(scale=0.5, size=(10, 2)) for c in centers])
    true_labels = np.repeat([0, 1, 2], 10)
    pred_labels = kmeans_cluster(X, k=3, seed=0)
    assert adjusted_rand_index(true_labels, pred_labels) == pytest.approx(1.0)


def test_loco_axis_accuracy_perfect_for_separated_levels():
    rng = np.random.default_rng(4)
    levels = np.array([0.0, 10.0, 20.0])
    axis_labels = np.repeat(levels, 3)
    X = axis_labels[:, None] + rng.normal(scale=0.2, size=(9, 1))
    assert loco_axis_accuracy(X, axis_labels) == pytest.approx(1.0)


def test_loco_axis_accuracy_below_one_when_levels_overlap():
    rng = np.random.default_rng(5)
    levels = np.array([0.0, 1.0, 2.0])
    axis_labels = np.repeat(levels, 3)
    X = axis_labels[:, None] + rng.normal(scale=5.0, size=(9, 1))
    assert loco_axis_accuracy(X, axis_labels) < 1.0


def _synthetic_grid_labels() -> dict[str, CaseGridLabel]:
    Pb_by_case = dict(zip(TRAIN_CASE_IDS, np.repeat([1.5, 2.0, 5.0], 3)))
    Thw_by_case = dict(zip(TRAIN_CASE_IDS, [1.1, 1.2, 1.4] * 3))
    labels = {
        c: CaseGridLabel(Pb_Pc=Pb_by_case[c], Thw_Tc=Thw_by_case[c]) for c in TRAIN_CASE_IDS
    }
    labels["case10"] = CaseGridLabel(Pb_Pc=1.5, Thw_Tc=1.185)
    labels["case15"] = CaseGridLabel(Pb_Pc=1.5, Thw_Tc=1.132)
    return labels


def test_evaluate_regime_discovery_recovers_a_clean_synthetic_signal():
    # A single feature tracking Pb_Pc almost exactly. This is a wiring
    # test (row order, ARI, LOCO, OOD nearest-centroid all line up
    # correctly), not a claim about the real DNS features — note a
    # second, purely-random column would NOT be a harmless no-op here:
    # standardize() gives it the same unit variance as the signal column,
    # so PCA can end up mixing a meaningful fraction of noise into PC1
    # (the exact per-feature-standardization dilution effect documented
    # in FINDINGS.md §2), which is a real phenomenon but not what this
    # test is checking.
    rng = np.random.default_rng(6)
    labels = _synthetic_grid_labels()
    X = np.empty((len(ALL_CASE_IDS), 1))
    for i, case_id in enumerate(ALL_CASE_IDS):
        X[i, 0] = labels[case_id].Pb_Pc + rng.normal(scale=0.01)

    result = evaluate_regime_discovery(X, labels)

    assert result["ari_vs_Pb_Pc"] == pytest.approx(1.0)
    assert result["loco_accuracy_Pb_Pc"] == pytest.approx(1.0)
    assert result["ood_nearest_Pb_Pc_centroid"]["case10"] == pytest.approx(1.5)
    assert result["ood_nearest_Pb_Pc_centroid"]["case15"] == pytest.approx(1.5)


def test_mfa_normalize_block_sets_leading_singular_value_to_one():
    rng = np.random.default_rng(7)
    X = rng.normal(loc=[10, -5, 0], scale=[5, 1, 20], size=(11, 3))
    Xn = mfa_normalize_block(X)
    _u, singular_values, _vt = np.linalg.svd(Xn - Xn.mean(axis=0), full_matrices=False)
    assert singular_values[0] == pytest.approx(1.0)


def test_mfa_normalize_block_handles_zero_variance_block():
    X = np.ones((10, 2))
    Xn = mfa_normalize_block(X)
    np.testing.assert_allclose(Xn, 0.0)


def test_combine_blocks_mfa_concatenates_columns():
    a = np.random.default_rng(0).normal(size=(6, 2))
    b = np.random.default_rng(1).normal(size=(6, 5))
    combined = combine_blocks_mfa({"a": a, "b": b})
    assert combined.shape == (6, 7)


def test_mfa_combination_recovers_a_signal_that_naive_concatenation_destroys():
    # A tiny "signal" block (2 features tracking Pb_Pc closely) next to a
    # much larger, unstructured "noise" block (40 pure-noise features) —
    # a scaled-up reproduction of the dilution FINDINGS.md §2/§3 found in
    # the real rich feature set (mean_profile's 384 features drowning out
    # bulk's 7). Naive concatenation + per-feature standardization should
    # destroy the signal outright (worse than the -1/3 to 0 "no better
    # than random" range); MFA block weighting should substantially
    # recover it — though not perfectly, since it only equalizes each
    # block's *leading* factor, not the aggregate noise floor across 40
    # unstructured dimensions (signal-only, with no noise block at all,
    # gets ARI 1.0 — MFA narrows the gap, it doesn't erase it).
    rng = np.random.default_rng(8)
    labels = _synthetic_grid_labels()
    n = len(ALL_CASE_IDS)

    signal = np.empty((n, 2))
    for i, case_id in enumerate(ALL_CASE_IDS):
        signal[i, 0] = labels[case_id].Pb_Pc + rng.normal(scale=0.01)
        signal[i, 1] = labels[case_id].Pb_Pc + rng.normal(scale=0.01)
    noise = rng.normal(size=(n, 40))

    naive = evaluate_regime_discovery(np.concatenate([signal, noise], axis=1), labels)
    combined = combine_blocks_mfa({"signal": signal, "noise": noise})
    weighted = evaluate_regime_discovery(combined, labels, standardize_input=False)

    assert naive["ari_vs_Pb_Pc"] < 0.1
    assert weighted["ari_vs_Pb_Pc"] > naive["ari_vs_Pb_Pc"] + 0.3

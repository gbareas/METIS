"""PCA, clustering, and cluster-agreement evaluation utilities for the
regime-discovery reference task (research_protocol.md).

Built on numpy/scipy only, deliberately not scikit-learn — pyproject.toml
reserves the "ml" extra (torch, scikit-learn, mlflow) for Phase 3+
(training, experiment tracking); PCA/k-means/ARI don't need it.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from metis.features.regime import ALL_CASE_IDS, OOD_CASE_IDS, TRAIN_CASE_IDS


def standardize(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score. A zero-variance column is left at zero rather
    than dividing by zero."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    return (X - mean) / std_safe


def pca(X: np.ndarray, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal component scores, explained-variance ratio, and loadings,
    via SVD of the (mean-centered) data matrix `X` (n_samples, n_features).

    Returns (scores, explained_variance_ratio, components), with
    `scores @ components` reconstructing the centered `X` exactly when
    `n_components` covers the full rank.
    """
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_components = n_components or S.shape[0]
    scores = U[:, :n_components] * S[:n_components]
    explained = (S**2) / np.sum(S**2)
    return scores, explained[:n_components], Vt[:n_components]


def kmeans_cluster(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means cluster assignment via scipy, with a fixed seed — a
    small-N regime-discovery result must be deterministic to be usable as
    evidence."""
    _centroids, labels = kmeans2(X, k, minit="++", seed=seed)
    return labels


def _comb2(n: np.ndarray) -> np.ndarray:
    return n * (n - 1) / 2


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand index between two labelings, via the contingency
    table (closed form) — 1.0 for identical labelings up to permutation,
    ~0 in expectation for independent/random labelings, negative for
    labelings that are systematically anti-correlated."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes_true, true_idx = np.unique(labels_true, return_inverse=True)
    classes_pred, pred_idx = np.unique(labels_pred, return_inverse=True)

    contingency = np.zeros((classes_true.size, classes_pred.size), dtype=np.int64)
    np.add.at(contingency, (true_idx, pred_idx), 1)

    sum_comb = _comb2(contingency).sum()
    sum_comb_c = _comb2(contingency.sum(axis=1)).sum()
    sum_comb_k = _comb2(contingency.sum(axis=0)).sum()

    n = labels_true.size
    total_comb = _comb2(np.array(n))
    expected_index = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected_index
    if denom == 0:
        return 1.0
    return float((sum_comb - expected_index) / denom)


def loco_axis_accuracy(X: np.ndarray, axis_labels: np.ndarray) -> float:
    """Leave-one-case-out nearest-centroid accuracy for one categorical
    axis (e.g. the Pb_Pc level or Thw_Tc level): for each sample, build
    per-level centroids from the *other* samples, assign the held-out
    sample to its nearest centroid, and report the fraction correctly
    assigned. Requires >= 2 samples per level (so a centroid still exists
    once the held-out sample is removed).
    """
    axis_labels = np.asarray(axis_labels)
    levels = np.unique(axis_labels)
    n = X.shape[0]
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        centroids = np.stack(
            [X[mask & (axis_labels == lvl)].mean(axis=0) for lvl in levels]
        )
        dists = np.linalg.norm(centroids - X[i], axis=1)
        predicted = levels[np.argmin(dists)]
        if predicted == axis_labels[i]:
            correct += 1
    return correct / n


def mfa_normalize_block(X: np.ndarray) -> np.ndarray:
    """Multiple-Factor-Analysis-style block weighting: standardize a
    feature block, then divide by its own leading singular value.

    This equalizes each block's leading-axis inertia to 1, so that once
    several blocks of very different raw dimensionality are concatenated,
    the block with the most features (or largest incidental scale) can't
    dominate the combined PCA purely by outnumbering the others — the
    exact failure mode FINDINGS.md §2/§3 diagnosed in naive concatenation
    (`mean_profile`'s 384 features drowning out `bulk`'s 7, independent of
    which one actually carries more physical signal).
    """
    Xz = standardize(X)
    _u, singular_values, _vt = np.linalg.svd(Xz - Xz.mean(axis=0), full_matrices=False)
    sigma1 = float(singular_values[0])
    if sigma1 == 0:
        return Xz
    return Xz / sigma1


def combine_blocks_mfa(blocks: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate feature blocks after MFA-style per-block weighting.

    `blocks`: block_name -> (n_samples, n_block_features) array, all
    sharing the same row order. The result is already appropriately
    scaled — pass `standardize_input=False` to `evaluate_regime_discovery`
    so its own per-feature standardization doesn't undo the block
    weighting (column-wise z-scoring would re-equalize every individual
    feature's variance again, exactly the effect this function exists to
    avoid).
    """
    return np.concatenate([mfa_normalize_block(X) for X in blocks.values()], axis=1)


def evaluate_regime_discovery(X: np.ndarray, labels: dict, standardize_input: bool = True) -> dict:
    """Standardize -> PCA -> k=3 cluster -> ARI/LOCO/OOD-nearest-centroid
    evaluation against the known (Pb_Pc, Thw_Tc) grid: the H1/H2/H3 test
    from research_protocol.md. Shared by every feature set/ablation block
    in scripts/run_regime_discovery.py and scripts/run_regime_ablation.py
    so they're all judged identically.

    `labels`: case_id -> metis.features.regime.CaseGridLabel, for every
    id in `metis.features.regime.ALL_CASE_IDS` (the row order `X` must
    match).

    `standardize_input`: set False when `X` already comes from
    `combine_blocks_mfa` (or is otherwise pre-scaled) — the default per-
    feature standardization would undo MFA block weighting.
    """
    Xz = standardize(X) if standardize_input else X
    scores, explained, _components = pca(Xz)

    train_idx = [ALL_CASE_IDS.index(c) for c in TRAIN_CASE_IDS]
    ood_idx = [ALL_CASE_IDS.index(c) for c in OOD_CASE_IDS]

    X_train = scores[train_idx]
    Pb_Pc_train = np.array([labels[c].Pb_Pc for c in TRAIN_CASE_IDS])
    Thw_Tc_train = np.array([labels[c].Thw_Tc for c in TRAIN_CASE_IDS])

    clusters = kmeans_cluster(X_train, k=3, seed=0)
    ari_pb = adjusted_rand_index(Pb_Pc_train, clusters)
    ari_thw = adjusted_rand_index(Thw_Tc_train, clusters)

    loco_pb = loco_axis_accuracy(X_train, Pb_Pc_train)
    loco_thw = loco_axis_accuracy(X_train, Thw_Tc_train)

    pb_centroids = {
        float(lvl): X_train[Pb_Pc_train == lvl].mean(axis=0) for lvl in np.unique(Pb_Pc_train)
    }
    ood_nearest_pb = {}
    for case_id, idx in zip(OOD_CASE_IDS, ood_idx):
        dists = {lvl: float(np.linalg.norm(scores[idx] - cen)) for lvl, cen in pb_centroids.items()}
        ood_nearest_pb[case_id] = min(dists, key=dists.get)

    return {
        "n_features": X.shape[1],
        "explained_variance_ratio": explained[:5].tolist(),
        "kmeans_clusters_train": clusters.tolist(),
        "Pb_Pc_train": Pb_Pc_train.tolist(),
        "Thw_Tc_train": Thw_Tc_train.tolist(),
        "ari_vs_Pb_Pc": ari_pb,
        "ari_vs_Thw_Tc": ari_thw,
        "loco_accuracy_Pb_Pc": loco_pb,
        "loco_accuracy_Thw_Tc": loco_thw,
        "ood_nearest_Pb_Pc_centroid": ood_nearest_pb,
        "ood_true_Pb_Pc": {c: labels[c].Pb_Pc for c in OOD_CASE_IDS},
        "ood_true_Thw_Tc": {c: labels[c].Thw_Tc for c in OOD_CASE_IDS},
    }

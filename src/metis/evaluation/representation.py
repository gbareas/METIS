"""Score a learned representation against the known regime axes
(milestone I2).

Same instruments as `metis.evaluation.regime` (k=3 clustering ARI, LOCO
nearest-centroid accuracy, OOD nearest-Pb_Pc-centroid) but applied to an
arbitrary latent `Z` instead of a PCA of raw features — so a PCA baseline
and an autoencoder latent are directly comparable, and the I2 stop
criterion ("does the nonlinear latent add anything over PCA?") can be
checked mechanically.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes

from metis.evaluation.metrics import relative_l2
from metis.evaluation.regime import (
    adjusted_rand_index,
    kmeans_cluster,
    loco_axis_accuracy,
)

_METRICS = ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc", "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc")


def evaluate_representation(
    Z_train: np.ndarray,
    Pb_Pc_train: np.ndarray,
    Thw_Tc_train: np.ndarray,
    *,
    Z_ood: np.ndarray | None = None,
    ood_case_ids: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Cluster/organisation metrics for a latent embedding of the training
    cases, plus OOD nearest-`Pb_Pc`-centroid placement if `Z_ood` given."""
    Z_train = np.asarray(Z_train, dtype=float)
    Pb_Pc_train = np.asarray(Pb_Pc_train)
    Thw_Tc_train = np.asarray(Thw_Tc_train)

    clusters = kmeans_cluster(Z_train, k=3, seed=seed)
    out = {
        "n_latent": int(Z_train.shape[1]),
        "ari_vs_Pb_Pc": adjusted_rand_index(Pb_Pc_train, clusters),
        "ari_vs_Thw_Tc": adjusted_rand_index(Thw_Tc_train, clusters),
        "loco_accuracy_Pb_Pc": loco_axis_accuracy(Z_train, Pb_Pc_train),
        "loco_accuracy_Thw_Tc": loco_axis_accuracy(Z_train, Thw_Tc_train),
    }

    if Z_ood is not None:
        Z_ood = np.asarray(Z_ood, dtype=float)
        ids = ood_case_ids or [f"ood{i}" for i in range(len(Z_ood))]
        centroids = {
            float(lvl): Z_train[Pb_Pc_train == lvl].mean(axis=0)
            for lvl in np.unique(Pb_Pc_train)
        }
        nearest = {}
        for cid, z in zip(ids, Z_ood):
            d = {lvl: float(np.linalg.norm(z - c)) for lvl, c in centroids.items()}
            nearest[cid] = min(d, key=d.get)
        out["ood_nearest_Pb_Pc_centroid"] = nearest

    return out


def latent_stability(latents: list[np.ndarray], *, reference: int = 0) -> dict:
    """How reproducible a latent embedding is across repeated fits (e.g.
    different seeds). Each latent is aligned to the reference by
    orthogonal Procrustes (rotation/reflection of latent axes is
    meaningless), then compared by relative L2. Returns the mean and max
    misalignment over the non-reference latents (0 = identical)."""
    if len(latents) < 2:
        return {"mean_rel_l2": 0.0, "max_rel_l2": 0.0, "n": len(latents)}
    ref = np.asarray(latents[reference], dtype=float)
    diffs = []
    for i, z in enumerate(latents):
        if i == reference:
            continue
        z = np.asarray(z, dtype=float)
        rot, _ = orthogonal_procrustes(z, ref)
        diffs.append(relative_l2(ref, z @ rot))
    return {
        "mean_rel_l2": float(np.mean(diffs)),
        "max_rel_l2": float(np.max(diffs)),
        "n": len(latents),
    }


def latent_physical_correlation(
    Z: np.ndarray, physical: dict[str, np.ndarray]
) -> dict[str, float]:
    """For each physical variable, the strongest |Pearson correlation|
    between it and any single latent axis — how directly the latent
    encodes that variable."""
    Z = np.asarray(Z, dtype=float)
    out = {}
    for name, values in physical.items():
        v = np.asarray(values, dtype=float)
        cors = [abs(np.corrcoef(Z[:, j], v)[0, 1]) for j in range(Z.shape[1])]
        out[name] = float(np.nanmax(cors))
    return out


def compare_to_baseline(
    candidate: dict, baseline: dict, *, margin: float = 1e-6
) -> dict:
    """Per-metric verdict of `candidate` vs `baseline` (both from
    `evaluate_representation`). `beats_baseline` is True only if the
    candidate strictly improves at least one clustering/LOCO metric by
    more than `margin` without regressing any other by more than `margin`."""
    deltas = {m: candidate[m] - baseline[m] for m in _METRICS if m in candidate and m in baseline}
    improved = [m for m, d in deltas.items() if d > margin]
    regressed = [m for m, d in deltas.items() if d < -margin]
    return {
        "deltas": deltas,
        "improved": improved,
        "regressed": regressed,
        "beats_baseline": bool(improved) and not regressed,
    }


# --- I2-B: pick a latent dimension from the PCA spectrum -------
def explained_variance_curve(singular_values) -> np.ndarray:
    """Cumulative reconstructed-variance fraction from a (descending)
    singular-value spectrum: `cumsum(s**2) / sum(s**2)`."""
    s2 = np.asarray(singular_values, dtype=float) ** 2
    return np.cumsum(s2) / np.sum(s2)


def choose_latent_dim(
    singular_values, grid, *, threshold: float = 0.90
) -> dict:
    """Smallest `k` in `grid` whose PCA reconstruction captures at least
    `threshold` of the variance. If none do, returns the largest `k` and
    `reached=False`."""
    cum = explained_variance_curve(singular_values)
    grid = sorted(int(k) for k in grid)
    for k in grid:
        if k <= cum.size and cum[k - 1] >= threshold:
            return {"headline_latent_dim": k, "reconstructed_variance": float(cum[k - 1]),
                    "threshold": threshold, "reached": True}
    k = grid[-1]
    return {"headline_latent_dim": k,
            "reconstructed_variance": float(cum[min(k, cum.size) - 1]),
            "threshold": threshold, "reached": False}

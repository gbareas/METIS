"""Per-case feature vectors for the regime-discovery reference task
(research_protocol.md primary task).

Two feature sets, both built from the standard physics layer:

- **compact** (`build_feature_vector`, 14 features) — the first-cut
  vector: 7 bulk dimensionless groups + 2 RMS-profile *peak* scalars +
  5 POD energy fractions from a single slice/field. See FINDINGS.md §1:
  this compact set only weakly recovers the pressure axis (H1) and
  carries no thermal-axis signal at all (H2).
- **rich** (`build_feature_vector_rich`, ~680 features) — the FINDINGS.md
  §1 enrichment: full wall-normal mean/RMS *profiles* (not just their
  peak) and POD energy fractions from multiple slice locations and both
  the velocity and temperature fields (not just one). All cases share the
  same 96x128x96 grid and NX=NZ=96 slice grid, so profile/spectrum-length
  features are directly comparable across cases without interpolation.

Both feed into `metis.evaluation.regime` (standardize -> PCA -> cluster)
identically; only the raw feature content differs. Keep both — the
compact set stays the fast/interpretable baseline the rich set is judged
against, not a superseded draft.

Expects the group's standard data layout, all under one `data_root`:
    data_root/raw/case{NN}/*.h5                       (HDF5Reader)
    data_root/processed/case{NN}/metadata.json         (resolved automatically
                                                         by HDF5Reader)
    data_root/processed_slices/case{NN}/{slice_id}/    (SliceReader)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metis.data.ingestion.hdf5_reader import DNSCase, HDF5Reader, latest_snapshot
from metis.data.ingestion.slice_reader import SliceReader
from metis.data.registry import CaseRegistry
from metis.features.physics import case_physics_summary, wall_normal_profiles
from metis.features.pod import pod

TRAIN_CASE_IDS = tuple(f"case{n:02d}" for n in range(1, 10))
OOD_CASE_IDS = ("case10", "case15")
ALL_CASE_IDS = TRAIN_CASE_IDS + OOD_CASE_IDS

BULK_FEATURE_NAMES = ("Re_b", "Pr_b", "Ec_b", "Br_b", "Ma_b", "Re_tau_cw", "Re_tau_hw")
POD_K = 5

# --- compact feature set -----------------------------------------------

COMPACT_POD_SLICE_ID = "s3_center"
COMPACT_POD_FIELD = "u"

FEATURE_NAMES = (
    list(BULK_FEATURE_NAMES)
    + ["rmsf_u_peak", "rmsf_T_peak"]
    + [f"pod_energy_frac_{i}" for i in range(POD_K)]
)

# --- rich feature set (FINDINGS.md §1 enrichment) -----------------------

RICH_MEAN_PROFILE_FIELDS = ("avg_u", "avg_T", "avg_rho")
RICH_RMS_PROFILE_FIELDS = ("rmsf_u", "rmsf_T")
RICH_POD_SLICE_IDS = ("s3_center", "s5_y_plus_10_cw", "s8_y_plus_10_hw")
RICH_POD_FIELDS = ("u", "T")


@dataclass
class CaseGridLabel:
    Pb_Pc: float
    Thw_Tc: float


def _pod_energy_fractions(data_root: Path, case_id: str, slice_id: str, field: str) -> np.ndarray:
    slice_case = SliceReader().read(data_root / "processed_slices" / case_id / slice_id)
    fractions = pod(slice_case, field).energy_fractions()
    assert fractions.shape[0] >= POD_K, (
        f"{case_id}/{slice_id}/{field}: need >= {POD_K} snapshots for POD "
        f"energy fractions, got {fractions.shape[0]}"
    )
    return fractions[:POD_K]


def _read_case(data_root: Path, case_id: str) -> DNSCase:
    return HDF5Reader().read(latest_snapshot(data_root / "raw" / case_id))


def build_feature_vector(case_id: str, data_root: Path) -> np.ndarray:
    """Compact (14-feature) vector for one case."""
    data_root = Path(data_root)
    case = _read_case(data_root, case_id)

    bulk = case_physics_summary(case)
    profiles = wall_normal_profiles(case, ("rmsf_u", "rmsf_T"))
    rmsf_u_peak = float(profiles["rmsf_u"][1:-1].max())
    rmsf_T_peak = float(profiles["rmsf_T"][1:-1].max())

    fractions = _pod_energy_fractions(data_root, case_id, COMPACT_POD_SLICE_ID, COMPACT_POD_FIELD)

    values = (
        [bulk[name] for name in BULK_FEATURE_NAMES]
        + [rmsf_u_peak, rmsf_T_peak]
        + list(fractions)
    )
    return np.array(values, dtype=np.float64)


def build_feature_matrix(case_ids: tuple[str, ...], data_root: Path) -> np.ndarray:
    """Stack `build_feature_vector` over `case_ids` -> (n_cases, n_features)."""
    return np.stack([build_feature_vector(c, data_root) for c in case_ids])


# --- feature blocks (FINDINGS.md §2 ablation) ---------------------------
#
# The rich set above concatenates all four blocks with equal per-feature
# standardization, which turned out to hand implicit "voting power" to
# whichever axis happens to modulate more individual features (FINDINGS.md
# §2). These per-block builders let each block be run through the same
# H1/H2/H3 pipeline in isolation to find out which one actually carries
# which signal.

FEATURE_BLOCK_NAMES = ("bulk", "mean_profile", "rms_profile", "pod")


def _bulk_items(case: DNSCase) -> list[tuple[str, float]]:
    bulk = case_physics_summary(case)
    return [(name, float(bulk[name])) for name in BULK_FEATURE_NAMES]


def _mean_profile_items(case: DNSCase) -> list[tuple[str, float]]:
    profiles = wall_normal_profiles(case, RICH_MEAN_PROFILE_FIELDS)
    items = []
    for name in RICH_MEAN_PROFILE_FIELDS:
        items += [(f"{name}_y{i}", float(v)) for i, v in enumerate(profiles[name][1:-1])]
    return items


def _rms_profile_items(case: DNSCase) -> list[tuple[str, float]]:
    profiles = wall_normal_profiles(case, RICH_RMS_PROFILE_FIELDS)
    items = []
    for name in RICH_RMS_PROFILE_FIELDS:
        items += [(f"{name}_y{i}", float(v)) for i, v in enumerate(profiles[name][1:-1])]
    return items


def _pod_items(data_root: Path, case_id: str) -> list[tuple[str, float]]:
    items = []
    for slice_id in RICH_POD_SLICE_IDS:
        for field in RICH_POD_FIELDS:
            fractions = _pod_energy_fractions(data_root, case_id, slice_id, field)
            items += [
                (f"pod_{slice_id}_{field}_frac{i}", float(v)) for i, v in enumerate(fractions)
            ]
    return items


def _block_items(
    block: str, case_id: str, data_root: Path, case: DNSCase | None = None
) -> list[tuple[str, float]]:
    if block == "pod":
        return _pod_items(data_root, case_id)
    if case is None:
        case = _read_case(data_root, case_id)
    if block == "bulk":
        return _bulk_items(case)
    if block == "mean_profile":
        return _mean_profile_items(case)
    if block == "rms_profile":
        return _rms_profile_items(case)
    raise ValueError(f"unknown feature block {block!r}, expected one of {FEATURE_BLOCK_NAMES}")


def build_feature_vector_block(case_id: str, data_root: Path, block: str) -> np.ndarray:
    """One feature block in isolation, for the per-block ablation."""
    data_root = Path(data_root)
    items = _block_items(block, case_id, data_root)
    return np.array([v for _, v in items], dtype=np.float64)


def block_feature_names(case_id: str, data_root: Path, block: str) -> list[str]:
    return [name for name, _ in _block_items(block, case_id, Path(data_root))]


def build_feature_matrix_block(
    case_ids: tuple[str, ...], data_root: Path, block: str
) -> tuple[np.ndarray, list[str]]:
    names = block_feature_names(case_ids[0], data_root, block)
    X = np.stack([build_feature_vector_block(c, data_root, block) for c in case_ids])
    return X, names


def _rich_feature_items(case_id: str, data_root: Path) -> list[tuple[str, float]]:
    data_root = Path(data_root)
    case = _read_case(data_root, case_id)
    items: list[tuple[str, float]] = []
    for block in ("bulk", "mean_profile", "rms_profile"):
        items += _block_items(block, case_id, data_root, case=case)
    items += _block_items("pod", case_id, data_root)
    return items


def build_feature_vector_rich(case_id: str, data_root: Path) -> np.ndarray:
    """Enriched (~680-feature) vector for one case — full mean/RMS
    profiles and multi-slice/multi-field POD content, per FINDINGS.md
    §1's proposed enrichment. Concatenates all of `FEATURE_BLOCK_NAMES`."""
    return np.array([v for _, v in _rich_feature_items(case_id, data_root)], dtype=np.float64)


def rich_feature_names(case_id: str, data_root: Path) -> list[str]:
    """Feature names for `build_feature_vector_rich`, in the same order.
    Derived from one case since all cases share the same grid/slice
    shapes and therefore the same feature layout."""
    return [name for name, _ in _rich_feature_items(case_id, data_root)]


def build_feature_matrix_rich(
    case_ids: tuple[str, ...], data_root: Path
) -> tuple[np.ndarray, list[str]]:
    """Stack `build_feature_vector_rich` over `case_ids`.

    Returns (X, feature_names) since the rich feature layout isn't a
    fixed module-level constant the way the compact `FEATURE_NAMES` is.
    """
    names = rich_feature_names(case_ids[0], data_root)
    X = np.stack([build_feature_vector_rich(c, data_root) for c in case_ids])
    return X, names


def case_grid_labels(
    source: Path | str | CaseRegistry, case_ids: tuple[str, ...]
) -> dict[str, CaseGridLabel]:
    """True (Pb_Pc, Thw_Tc) operating condition per case — the ground
    truth regime discovery is benchmarked against.

    `source` is either a data root (path) or a `CaseRegistry`; with a
    registry the descriptors carry the ratios, so no metadata file is
    re-parsed here.
    """
    if isinstance(source, CaseRegistry):
        return {
            c: CaseGridLabel(Pb_Pc=source[c].Pb_Pc, Thw_Tc=source[c].Thw_Tc)
            for c in case_ids
        }
    data_root = Path(source)
    labels = {}
    for case_id in case_ids:
        meta = json.loads((data_root / "processed" / case_id / "metadata.json").read_text())
        labels[case_id] = CaseGridLabel(Pb_Pc=meta["Pb_Pc"], Thw_Tc=meta["Thw_Tc"])
    return labels

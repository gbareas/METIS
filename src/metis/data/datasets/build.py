"""Build (and cache) Level-1 feature datasets from the case registry
(milestone R5).

`build_feature_dataset(...)` extracts one of the regime feature sets for a
list of cases and returns a `FeatureDataset`. If `out_dir` already holds
an artifact whose fingerprint matches the request, it is loaded instead
of recomputed — so re-running an evaluation does not re-read raw DNS
unless the cases, the feature set, or the feature-extraction code changed.
It does NOT detect edits to the underlying DNS files; pass
`rebuild=True` (or delete the artifact) if those change.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from metis.config import git_commit
from metis.data.datasets.feature_dataset import FeatureDataset
from metis.data.registry import CaseRegistry
from metis.features import regime
from metis.features.regime import (
    FEATURE_BLOCK_NAMES,
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_matrix_block,
    build_feature_matrix_rich,
)

# Source files whose content defines what a feature value *is* — a change
# to any of these must invalidate cached datasets.
_FEATURE_SOURCE_MODULES = ("regime", "physics", "pod")

FEATURE_SETS = ("compact", "rich", *FEATURE_BLOCK_NAMES)


def _build_matrix(feature_set: str, case_ids, data_root):
    if feature_set == "compact":
        return build_feature_matrix(tuple(case_ids), data_root), list(FEATURE_NAMES)
    if feature_set == "rich":
        X, names = build_feature_matrix_rich(tuple(case_ids), data_root)
        return X, list(names)
    if feature_set in FEATURE_BLOCK_NAMES:
        X, names = build_feature_matrix_block(tuple(case_ids), data_root, feature_set)
        return X, list(names)
    raise ValueError(f"unknown feature_set {feature_set!r}, expected one of {FEATURE_SETS}")


def _feature_code_hash() -> str:
    h = hashlib.sha256()
    features_dir = Path(regime.__file__).resolve().parent
    for name in sorted(_FEATURE_SOURCE_MODULES):
        h.update((features_dir / f"{name}.py").read_bytes())
    return h.hexdigest()


def _fingerprint(feature_set: str, case_ids) -> str:
    payload = json.dumps(
        {
            "feature_set": feature_set,
            "case_ids": list(case_ids),
            "feature_code": _feature_code_hash(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_feature_dataset(
    registry: CaseRegistry,
    case_ids,
    feature_set: str = "compact",
    *,
    out_dir: str | Path | None = None,
    rebuild: bool = False,
) -> FeatureDataset:
    """Extract `feature_set` for `case_ids`; cache to / load from `out_dir`."""
    case_ids = list(case_ids)
    fp = _fingerprint(feature_set, case_ids)

    if out_dir is not None and not rebuild and FeatureDataset.exists_at(out_dir):
        cached = FeatureDataset.load(out_dir)
        if cached.fingerprint == fp:
            return cached

    X, feature_names = _build_matrix(feature_set, case_ids, registry.data_root)
    dataset = FeatureDataset(
        X=X,
        case_ids=case_ids,
        feature_names=feature_names,
        feature_set=feature_set,
        provenance={
            "fingerprint": fp,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": git_commit(),
            "data_root": str(registry.data_root),
            "n_cases": len(case_ids),
            "n_features": int(X.shape[1]),
        },
    )
    if out_dir is not None:
        dataset.save(out_dir)
    return dataset

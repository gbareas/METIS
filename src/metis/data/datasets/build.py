"""Build (and cache) Level-1 feature datasets from the case registry
(milestone R5; fingerprinting hardened per updated-plan §20).

`build_feature_dataset(...)` extracts one of the regime feature sets for a
list of cases and returns a `FeatureDataset`. If `out_dir` already holds
an artifact whose fingerprint matches the request, it is loaded instead
of recomputed.

The fingerprint covers everything that can change a feature value:

  - the feature set and case ids;
  - the source of the feature-extraction + ingestion + preprocessing
    code (`features/{regime,physics,pod,spectra}.py`,
    `data/ingestion/*.py`, `data/preprocessing/*.py`);
  - a lightweight **source manifest** — for every raw / processed / slice
    file of every requested case: relative path, size, mtime, and (for
    the small metadata files) a content hash.

TB-scale field files are *not* hashed byte-by-byte; size + mtime is the
reliable-invalidation signal. `rebuild=True` forces a rebuild regardless.
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

# Source files whose content defines what a feature value *is*.
_FEATURE_SOURCES = (
    "features/regime.py", "features/physics.py", "features/pod.py", "features/spectra.py",
    "data/ingestion/hdf5_reader.py", "data/ingestion/slice_reader.py",
    "data/preprocessing/base.py", "data/preprocessing/scalers.py",
    "data/preprocessing/fields.py",
)

FEATURE_SETS = ("compact", "rich", *FEATURE_BLOCK_NAMES)
_METIS_ROOT = Path(regime.__file__).resolve().parents[1]  # src/metis/


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


def _code_hash() -> str:
    h = hashlib.sha256()
    for rel in _FEATURE_SOURCES:
        p = _METIS_ROOT / rel
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def _source_manifest(registry: CaseRegistry | None, case_ids) -> str:
    """Hash of (path, size, mtime, [content hash for small metadata]) over
    every input file of every requested case. Empty when no registry."""
    if registry is None:
        return "no-registry"
    entries: list = []
    for cid in case_ids:
        try:
            desc = registry[cid]
        except KeyError:
            entries.append([cid, "missing"])
            continue
        for f in sorted(desc.raw_dir.glob("*.h5")) if desc.raw_dir.is_dir() else []:
            st = f.stat()
            entries.append([f"raw/{cid}/{f.name}", st.st_size, st.st_mtime_ns])
        if desc.metadata_path.exists():
            raw = desc.metadata_path.read_bytes()
            entries.append([f"processed/{cid}/metadata.json", len(raw),
                            hashlib.sha256(raw).hexdigest()[:16]])
        if desc.slice_root and desc.slice_root.is_dir():
            for f in sorted(desc.slice_root.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    entries.append([f"slices/{cid}/{f.relative_to(desc.slice_root)}",
                                    st.st_size, st.st_mtime_ns])
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint(feature_set: str, case_ids, registry: CaseRegistry | None = None) -> str:
    payload = json.dumps(
        {
            "feature_set": feature_set,
            "case_ids": list(case_ids),
            "code_hash": _code_hash(),
            "source_manifest": _source_manifest(registry, case_ids),
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
    fp = _fingerprint(feature_set, case_ids, registry)

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

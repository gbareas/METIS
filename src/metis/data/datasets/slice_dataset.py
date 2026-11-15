"""Level-2 slice-snapshot dataset (I2-B / updated-plan §26).

One 2-D fluctuation field per snapshot as a sample — thousands of them,
unlike the Level-1 one-row-per-case `FeatureDataset`. Built for the I2-B
representation study (`docs/i2b_representation_protocol.md`); stored as
three `.npz` splits + a `metadata.json` under one directory::

    <dir>/train.npz  X (n, NX, NZ) float32, case_ids (n,), snapshot_idx (n,)
    <dir>/val.npz
    <dir>/ood.npz
    <dir>/metadata.json  field, slice_id, grid, splits, scaler {mean,std}, provenance

The stored fields are **raw** fluctuations; `metadata["scaler"]` (a
single global mean/std fit on the *train* split only) is how to
standardise. `SliceDataset.load(dir, split)` returns one split with
`.standardize()` / `.inverse()` helpers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metis.config import git_commit
from metis.data.datasets.build import _code_hash
from metis.data.registry import CaseRegistry

_SPLITS = ("train", "val", "ood")
METADATA_FILE = "metadata.json"


# --------------------------------------------------------------------- #
@dataclass
class SliceDataset:
    X: np.ndarray                 # (n, NX, NZ) raw fluctuation snapshots
    case_ids: np.ndarray          # (n,) str — which case each sample came from
    snapshot_idx: np.ndarray      # (n,) int — its index within that case's slice
    field: str
    slice_id: str
    split: str
    scaler: dict                  # {"mean": float, "std": float}, fit on train
    provenance: dict

    def __len__(self) -> int:
        return len(self.X)

    def standardize(self, X=None) -> np.ndarray:
        X = self.X if X is None else X
        return (np.asarray(X, dtype=np.float32) - self.scaler["mean"]) / self.scaler["std"]

    def inverse(self, Xs) -> np.ndarray:
        return np.asarray(Xs, dtype=np.float32) * self.scaler["std"] + self.scaler["mean"]

    @classmethod
    def load(cls, out_dir: str | Path, split: str) -> SliceDataset:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / METADATA_FILE).read_text())
        with np.load(out_dir / f"{split}.npz", allow_pickle=False) as npz:
            return cls(
                X=npz["X"], case_ids=npz["case_ids"], snapshot_idx=npz["snapshot_idx"],
                field=meta["field"], slice_id=meta["slice_id"], split=split,
                scaler=meta["scaler"], provenance=meta["provenance"],
            )


# --------------------------------------------------------------------- #
def normalize_split_config(cfg: dict) -> dict:
    """Turn the yaml `splits.<name>` shape into
    `{split: {"cases": [...], "snapshots": (start, stop)}}`."""
    tc = list(cfg["train_cases"])
    return {
        "train": {"cases": tc, "snapshots": tuple(cfg["train_snapshots"])},
        "val": {"cases": tc, "snapshots": tuple(cfg["val_snapshots"])},
        "ood": {"cases": list(cfg["ood_cases"]), "snapshots": tuple(cfg["ood_snapshots"])},
    }


def _source_manifest(registry: CaseRegistry, field: str, slice_id: str, splits: dict) -> str:
    cases = sorted({c for s in splits.values() for c in s["cases"]})
    entries = []
    for cid in cases:
        try:
            d = registry[cid]
        except KeyError:
            entries.append([cid, "missing"]); continue
        sdir = None if d.slice_root is None else d.slice_root / slice_id
        for name in (f"snapshots_{field}.npy", f"mean_{field}.npy", "metadata.json", "grid.npz"):
            p = None if sdir is None else sdir / name
            if p is not None and p.exists():
                st = p.stat()
                entries.append([f"{cid}/{slice_id}/{name}", st.st_size, st.st_mtime_ns])
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint(field: str, slice_id: str, splits: dict, manifest: str) -> str:
    payload = json.dumps(
        {"field": field, "slice_id": slice_id, "splits": splits,
         "code_hash": _code_hash(), "source_manifest": manifest},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _gather(registry: CaseRegistry, field: str, slice_id: str, spec: dict):
    lo, hi = spec["snapshots"]
    xs, cids, idxs = [], [], []
    for cid in spec["cases"]:
        sc = registry[cid].load_slice(slice_id, fields=(field,))
        snaps = np.asarray(sc.snapshots[field], dtype=np.float32)[lo:hi]
        xs.append(snaps)
        cids.append(np.full(len(snaps), cid))
        idxs.append(np.arange(lo, lo + len(snaps)))
    return (np.concatenate(xs), np.concatenate(cids), np.concatenate(idxs))


def build_slice_dataset(
    registry: CaseRegistry,
    field: str,
    slice_id: str,
    splits: dict,
    *,
    out_dir: str | Path,
    rebuild: bool = False,
) -> dict[str, SliceDataset]:
    """Build (or load from cache) the three splits. `splits` is the
    normalised form from `normalize_split_config`. Returns
    `{"train": ..., "val": ..., "ood": ...}`."""
    out_dir = Path(out_dir)
    manifest = _source_manifest(registry, field, slice_id, splits)
    fp = _fingerprint(field, slice_id, splits, manifest)

    meta_path = out_dir / METADATA_FILE
    if not rebuild and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached.get("provenance", {}).get("fingerprint") == fp and all(
            (out_dir / f"{s}.npz").exists() for s in _SPLITS
        ):
            return {s: SliceDataset.load(out_dir, s) for s in _SPLITS}

    raw = {s: _gather(registry, field, slice_id, splits[s]) for s in _SPLITS}
    train_X = raw["train"][0]
    scaler = {"mean": float(train_X.mean()), "std": float(train_X.std()) or 1.0}

    out_dir.mkdir(parents=True, exist_ok=True)
    grid = list(train_X.shape[1:])
    for s, (X, cids, idxs) in raw.items():
        np.savez_compressed(out_dir / f"{s}.npz", X=X, case_ids=cids, snapshot_idx=idxs)
    provenance = {
        "fingerprint": fp,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "data_root": str(registry.data_root),
        "n": {s: len(raw[s][0]) for s in _SPLITS},
    }
    meta_path.write_text(json.dumps(
        {"field": field, "slice_id": slice_id, "grid": grid, "splits": splits,
         "scaler": scaler, "provenance": provenance}, indent=2,
    ))
    return {s: SliceDataset.load(out_dir, s) for s in _SPLITS}

"""Ingestion for the homogeneous-plane (XZ) DNS slice snapshot product.

Local 3D DNS storage (see hdf5_reader.py) only holds 1-2 full-field
snapshots per case — not a dense time series — so spectra and POD/SPOD
can't be built on it directly. The group's slice-extraction pipeline
(shared/scripts/fetch_slice.py, already run for all 11 cases as part of
the pub5_neural_operators work) fills that gap: it stores time-resolved
2D-plane snapshots, which is exactly the multi-snapshot data spectral and
modal analysis need. This is a shared group data product, not pub5-specific
code — this reader treats `data/processed_slices/` as a first-class metis
input, independent of pub5's own code.

Layout:
    data/processed_slices/case{NN}/{slice_id}/
        snapshots_{field}.npy   (n_snapshots, NX, NZ) float32 FLUCTUATION
                                 fields (verified: snapshot time-mean is
                                 ~0.03 vs an O(1) mean field — the mean has
                                 already been subtracted upstream).
        mean_{field}.npy        (NX, NZ) time-mean field, stored separately.
        grid.npz                x (NX,), z (NZ,), y_loc () [metres] — the
                                 plane sits at a fixed wall-normal location.
        metadata.json           case + slice descriptors (Pb_Pc, Thw_Tc,
                                 Tcw_Tc, grid, n_snapshots, slice_prefix).

Scope: XZ (homogeneous-plane) slices only, identified by
`metadata["slice_prefix"]` starting with "plane_XZ_" — x and z are the
periodic/statistically-homogeneous directions spectra and POD/SPOD need.
The wall-normal (XY/ZY) slices use a different, less regular
metadata/grid schema (`grid: {"orientation", "rows", "cols"}` instead of
`{"NX", "NZ"}`) and aren't needed yet; add a separate reader for them if a
wall-normal-plane analysis is actually required later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIELDS = ("u", "T", "cp")

REQUIRED_SLICE_METADATA_KEYS = ("case", "slice_id", "Pb_Pc", "Thw_Tc", "Tcw_Tc", "grid")


@dataclass
class SliceCaseMetadata:
    case_id: str
    slice_id: str
    Pb_Pc: float
    Thw_Tc: float
    Tcw_Tc: float
    y_loc: float
    grid: dict[str, int]
    extra: dict = field(default_factory=dict)


@dataclass
class SliceCase:
    metadata: SliceCaseMetadata
    coordinates: dict[str, np.ndarray]  # {"x": (NX,), "z": (NZ,)}
    snapshots: dict[str, np.ndarray]  # field -> (n_snapshots, NX, NZ) fluctuation
    mean: dict[str, np.ndarray]  # field -> (NX, NZ)

    def variable_names(self) -> list[str]:
        return sorted(self.snapshots.keys())


class SliceReader:
    """Reads one case/slice_id directory of XZ homogeneous-plane snapshots
    into a `SliceCase`."""

    def __init__(self, fields: tuple[str, ...] = FIELDS):
        self.fields = fields

    def read(self, slice_dir: str | Path) -> SliceCase:
        slice_dir = Path(slice_dir)
        meta_path = slice_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Slice metadata not found: {meta_path}")
        case_meta = json.loads(meta_path.read_text())

        prefix = case_meta.get("slice_prefix", "")
        if not prefix.startswith("plane_XZ_"):
            raise ValueError(
                f"{slice_dir}: SliceReader only supports XZ (homogeneous-plane) "
                f"slices, got slice_prefix={prefix!r}"
            )

        grid_path = slice_dir / "grid.npz"
        if not grid_path.exists():
            raise FileNotFoundError(f"Slice grid not found: {grid_path}")
        grid = np.load(grid_path)
        coordinates = {"x": np.asarray(grid["x"]), "z": np.asarray(grid["z"])}

        snapshots: dict[str, np.ndarray] = {}
        mean: dict[str, np.ndarray] = {}
        for name in self.fields:
            snap_path = slice_dir / f"snapshots_{name}.npy"
            if not snap_path.exists():
                continue
            snapshots[name] = np.load(snap_path)
            mean[name] = np.load(slice_dir / f"mean_{name}.npy")

        metadata = self._read_metadata(case_meta, float(grid["y_loc"]))
        self._validate(metadata, snapshots)
        return SliceCase(metadata=metadata, coordinates=coordinates, snapshots=snapshots, mean=mean)

    def _read_metadata(self, case_meta: dict, y_loc: float) -> SliceCaseMetadata:
        missing = [k for k in REQUIRED_SLICE_METADATA_KEYS if k not in case_meta]
        if missing:
            raise ValueError(f"Slice metadata missing required keys: {missing}")
        extra = {k: v for k, v in case_meta.items() if k not in REQUIRED_SLICE_METADATA_KEYS}
        return SliceCaseMetadata(
            case_id=case_meta["case"],
            slice_id=case_meta["slice_id"],
            Pb_Pc=case_meta["Pb_Pc"],
            Thw_Tc=case_meta["Thw_Tc"],
            Tcw_Tc=case_meta["Tcw_Tc"],
            y_loc=y_loc,
            grid=case_meta["grid"],
            extra=extra,
        )

    def _validate(self, metadata: SliceCaseMetadata, snapshots: dict[str, np.ndarray]) -> None:
        if not snapshots:
            raise ValueError(f"{metadata.case_id}/{metadata.slice_id}: no snapshot fields found")
        shapes = {v.shape for v in snapshots.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"{metadata.case_id}/{metadata.slice_id}: inconsistent snapshot shapes: "
                f"{ {k: v.shape for k, v in snapshots.items()} }"
            )

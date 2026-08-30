"""Deliberately corrupt a generated mock case, to exercise the validation
layer (milestone R3). Each function mutates files produced by
`metis.testing.mock_dns` / `metis.testing.mock_slices` in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def inject_nan(h5_path: str | Path, field: str = "T", n: int = 3) -> None:
    """Set `n` cells of `field` in a snapshot HDF5 to NaN."""
    with h5py.File(h5_path, "r+") as f:
        arr = np.asarray(f[field])
        flat = arr.reshape(-1)
        flat[:n] = np.nan
        f[field][...] = flat.reshape(arr.shape)


def make_nonpositive(h5_path: str | Path, field: str = "rho") -> None:
    """Force `field` to contain a non-positive value (physically invalid
    for density/temperature/pressure/...)."""
    with h5py.File(h5_path, "r+") as f:
        arr = np.asarray(f[field])
        arr.reshape(-1)[0] = -abs(arr.reshape(-1)[0]) - 1.0
        f[field][...] = arr


def break_grid_metadata(
    meta_path: str | Path, *, Nx: int | None = None, Ny: int | None = None,
    Nz: int | None = None,
) -> None:
    """Overwrite grid dimensions in metadata.json so they no longer match
    the actual field shapes."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    for key, val in (("Nx", Nx), ("Ny", Ny), ("Nz", Nz)):
        if val is not None:
            meta["grid"][key] = val
    meta_path.write_text(json.dumps(meta, indent=2))


def unsort_timesteps(meta_path: str | Path) -> None:
    """Make metadata 'timesteps' non-ascending."""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    ts = list(meta.get("timesteps") or [1, 2, 3])
    meta["timesteps"] = list(reversed(ts)) if len(ts) > 1 else [3, 1, 2]
    meta_path.write_text(json.dumps(meta, indent=2))


def duplicate_slice_frame(
    slice_dir: str | Path, field: str = "u", src: int = 0, dst: int = 1
) -> None:
    """Overwrite snapshot frame `dst` with a copy of frame `src`."""
    path = Path(slice_dir) / f"snapshots_{field}.npy"
    arr = np.load(path)
    arr[dst] = arr[src]
    np.save(path, arr)


def inject_slice_nan(slice_dir: str | Path, field: str = "T") -> None:
    path = Path(slice_dir) / f"snapshots_{field}.npy"
    arr = np.load(path)
    arr.reshape(-1)[:3] = np.nan
    np.save(path, arr)

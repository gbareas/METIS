"""Generate a small synthetic XZ homogeneous-plane slice case for local
development/testing, mirroring the real
data/processed_slices/case{NN}/{slice_id}/ layout (see
metis.data.ingestion.slice_reader).

This is NOT DNS data — it exists so the slice ingestion layer and the
spectra/POD modules built on it can run and be tested end-to-end without
the real (group-internal) slice files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIELDS = ("u", "T", "cp")
_SCALE = {"u": 0.06, "T": 0.5, "cp": 50.0}
_MEAN = {"u": 1.0, "T": 300.0, "cp": 3000.0}


def generate(
    slice_dir: str | Path,
    nx: int = 16,
    nz: int = 16,
    n_snapshots: int = 32,
    seed: int = 0,
    case_id: str = "case_mock",
    slice_id: str = "s_mock",
    Pb_Pc: float = 1.5,
    Thw_Tc: float = 1.1,
    Tcw_Tc: float = 0.95,
    y_loc: float = 1.75e-4,
) -> Path:
    slice_dir = Path(slice_dir)
    slice_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 1.25e-3, nx, endpoint=False)
    z = np.linspace(0, 4.17e-4, nz, endpoint=False)
    np.savez(slice_dir / "grid.npz", x=x, z=z, y_loc=y_loc)

    for name in FIELDS:
        snapshots = rng.normal(0.0, _SCALE[name], (n_snapshots, nx, nz)).astype(np.float32)
        np.save(slice_dir / f"snapshots_{name}.npy", snapshots)
        mean_field = np.full((nx, nz), _MEAN[name], dtype=np.float32)
        np.save(slice_dir / f"mean_{name}.npy", mean_field)

    metadata = {
        "case": case_id,
        "slice_id": slice_id,
        "slice_prefix": f"plane_XZ_{slice_id}",
        "Pb_Pc": Pb_Pc,
        "Thw_Tc": Thw_Tc,
        "Tcw_Tc": Tcw_Tc,
        "grid": {"NX": nx, "NZ": nz},
        "n_snapshots": n_snapshots,
    }
    (slice_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return slice_dir

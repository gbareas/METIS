"""Generate a small synthetic DNS-like HDF5 snapshot for local development.

This is NOT DNS data — it exists so the ingestion layer (and everything
built on top of it) can run and be tested end-to-end without needing the
real (multi-GB, group-internal) RHEA output. The schema mirrors the real
solver's snapshot files as closely as practical: flat HDF5 layout (no
groups), [z, y, x] axis order, ghost cells, instantaneous/avg_/rmsf_/
favre_ field families, 3D coordinate arrays, and a companion metadata.json
carrying the case-level physics parameters that RHEA itself doesn't write
into the HDF5 file.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _with_ghosts(interior: np.ndarray) -> np.ndarray:
    """Pad a 1D interior coordinate array with one mirror-extrapolated
    ghost cell per side, matching RHEA's (N+2,) convention."""
    lo = interior[0] - (interior[1] - interior[0])
    hi = interior[-1] + (interior[-1] - interior[-2])
    return np.concatenate([[lo], interior, [hi]])


def generate(
    path: str | Path,
    nx: int = 8,
    ny: int = 16,
    nz: int = 8,
    seed: int = 0,
    case_id: str | None = None,
    Pb_Pc: float = 1.5,
    Thw_Tc: float = 1.1,
    Tcw_Tc: float = 0.95,
    iteration: int = 25_000_000,
    time: float = 0.025,
    averaging_time: float = 0.01,
) -> tuple[Path, Path]:
    """Write a mock snapshot HDF5 file plus its companion metadata.json.

    Returns (h5_path, metadata_path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    case_id = case_id or path.stem

    x1d = _with_ghosts(np.linspace(0, 1, nx))
    y1d = _with_ghosts(np.tanh(np.linspace(-2, 2, ny)))  # wall-clustered, like a channel mesh
    z1d = _with_ghosts(np.linspace(0, 1, nz))
    # RHEA writes arrays in [z, y, x] axis order (see hdf5_reader).
    shape = (nz + 2, ny + 2, nx + 2)

    z = np.tile(z1d[:, None, None], (1, ny + 2, nx + 2))
    y = np.tile(y1d[None, :, None], (nz + 2, 1, nx + 2))
    x = np.tile(x1d[None, None, :], (nz + 2, ny + 2, 1))

    instantaneous = {
        "rho": rng.normal(500.0, 50.0, shape),
        "u": rng.normal(1.0, 0.1, shape),
        "v": rng.normal(0.0, 0.05, shape),
        "w": rng.normal(0.0, 0.05, shape),
        "T": rng.normal(300.0, 20.0, shape),
        "P": rng.normal(8.0e6, 1.0e4, shape),
        "mu": rng.normal(3.0e-5, 1.0e-6, shape),
        "kappa": rng.normal(0.1, 0.01, shape),
        "c_p": rng.normal(2500.0, 200.0, shape),
        "sos": rng.normal(300.0, 10.0, shape),
    }
    fields: dict[str, np.ndarray] = dict(instantaneous)
    for name, values in instantaneous.items():
        fields[f"avg_{name}"] = values + rng.normal(0.0, 0.01 * np.abs(values).mean(), shape)
    # Mass-/energy-flux-weighted products RHEA also writes out (needed for
    # bulk-quantity extraction — see metis.features.physics).
    fields["avg_rhou"] = fields["avg_rho"] * fields["avg_u"]
    fields["avg_rhoT"] = fields["avg_rho"] * fields["avg_T"]
    fields["rmsf_u"] = np.abs(rng.normal(0.02, 0.005, shape))
    fields["rmsf_T"] = np.abs(rng.normal(0.5, 0.1, shape))
    fields["favre_uffuff"] = rng.normal(0.0, 0.01, shape)
    fields["favre_uffvff"] = rng.normal(0.0, 0.005, shape)
    fields["tag_IBM"] = np.zeros(shape)

    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("z", data=z)
        for name, values in fields.items():
            f.create_dataset(name, data=values)
        f.attrs["Iteration"] = [iteration]
        f.attrs["Time"] = [time]
        f.attrs["AveragingTime"] = [averaging_time]

    metadata_path = path.parent / "metadata.json"
    metadata = {
        "case": case_id,
        "case_number": 0,
        "Pb_Pc": Pb_Pc,
        "Thw_Tc": Thw_Tc,
        "Tcw_Tc": Tcw_Tc,
        "grid": {"Nx": nx, "Ny": ny, "Nz": nz},
        "n_snapshots": 1,
        "timesteps": [iteration],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return path, metadata_path

"""Ingestion layer for raw DNS output.

Reads the group's native RHEA HDF5 DNS files into a standardized, validated
in-memory representation (`DNSCase`). This is the one place in the
framework that should ever need to know the raw file layout — everything
downstream (preprocessing, statistics, POD/SPOD, ML datasets) consumes
`DNSCase` instead of touching HDF5 directly.

Real RHEA layout (verified against the group's data/raw/case*/*.h5 files):

    One HDF5 file = one snapshot/iteration (a case directory holds several,
    one per iteration checkpointed). Flat layout, no groups — every variable
    is a root-level dataset shaped (Nz+2, Ny+2, Nx+2): RHEA writes arrays in
    [z, y, x] axis order, with one ghost cell per side. Variable families
    present: instantaneous (u, v, w, T, P, rho, mu, kappa, c_p, c_v, sos,
    E), running time-average (avg_*), rms fluctuation (rmsf_*), Favre
    stresses (favre_*), an immersed-boundary mask (tag_IBM), and
    coordinates x/y/z as full 3D meshgrid arrays (constant along the other
    two axes — z varies along axis 0, y along axis 1, x along axis 2).
    Root attrs carry only Iteration/Time/AveragingTime — no case-level
    physics metadata.

    Case-level physics metadata (Pb_Pc, Thw_Tc, Tcw_Tc, grid, snapshot
    timesteps) lives in a companion JSON file, not in the HDF5 file itself:
    data/processed/case{NN}/metadata.json, sibling to data/raw/case{NN}/.

A snapshot file has no time axis — time series come from reading multiple
files in a case directory, or from the slice-extraction pipeline (see
pub5_neural_operators) for high-frequency 2D planes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# Root-level datasets that are coordinates, not physical fields.
COORDINATE_NAMES = ("x", "y", "z")

REQUIRED_CASE_METADATA_KEYS = ("case", "Pb_Pc", "Thw_Tc", "Tcw_Tc", "grid")


@dataclass
class DNSCaseMetadata:
    case_id: str
    Pb_Pc: float
    Thw_Tc: float
    Tcw_Tc: float
    grid: dict[str, int]  # {"Nx": .., "Ny": .., "Nz": ..}
    iteration: int
    time: float
    averaging_time: float
    extra: dict = field(default_factory=dict)


@dataclass
class DNSCase:
    metadata: DNSCaseMetadata
    coordinates: dict[str, np.ndarray]  # 1D profiles: {"x": ..., "y": ..., "z": ...}
    fields: dict[str, np.ndarray]  # every other root-level dataset

    def variable_names(self) -> list[str]:
        return sorted(self.fields.keys())


def latest_snapshot(case_dir: str | Path) -> Path:
    """Pick the snapshot with the largest iteration number (longest
    running-average window) in a raw case directory."""
    case_dir = Path(case_dir)
    files = sorted(case_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {case_dir}")
    return max(files, key=lambda p: int(re.search(r"_(\d+)\.h5$", p.name).group(1)))


class HDF5Reader:
    """Reads a single native RHEA DNS snapshot HDF5 file into a `DNSCase`."""

    def __init__(self, required_variables: list[str] | None = None):
        self.required_variables = required_variables

    def read(self, path: str | Path, metadata_path: str | Path | None = None) -> DNSCase:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DNS file not found: {path}")

        meta_path = Path(metadata_path) if metadata_path else self._default_metadata_path(path)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Case metadata not found: {meta_path} (pass metadata_path= "
                f"explicitly if this case doesn't live under data/raw/<case>/)"
            )
        case_meta = json.loads(meta_path.read_text())

        with h5py.File(path, "r") as f:
            metadata = self._read_metadata(f, case_meta)
            # RHEA arrays are [z, y, x]: z along axis 0, y along axis 1,
            # x along axis 2. Each coord dataset is a full meshgrid but
            # constant along the other two axes.
            coordinates = {
                "z": np.asarray(f["z"][:, 1, 1]),
                "y": np.asarray(f["y"][1, :, 1]),
                "x": np.asarray(f["x"][1, 1, :]),
            }
            fields = {
                name: np.asarray(f[name])
                for name in f
                if name not in COORDINATE_NAMES
            }

        self._validate(metadata, fields)
        return DNSCase(metadata=metadata, coordinates=coordinates, fields=fields)

    @staticmethod
    def _default_metadata_path(raw_path: Path) -> Path:
        """data/raw/case{NN}/snapshot.h5 -> data/processed/case{NN}/metadata.json"""
        case_dir = raw_path.parent
        return case_dir.parent.parent / "processed" / case_dir.name / "metadata.json"

    def _read_metadata(self, f: h5py.File, case_meta: dict) -> DNSCaseMetadata:
        missing = [k for k in REQUIRED_CASE_METADATA_KEYS if k not in case_meta]
        if missing:
            raise ValueError(f"Case metadata missing required keys: {missing}")
        extra = {k: v for k, v in case_meta.items() if k not in REQUIRED_CASE_METADATA_KEYS}
        return DNSCaseMetadata(
            case_id=case_meta["case"],
            Pb_Pc=case_meta["Pb_Pc"],
            Thw_Tc=case_meta["Thw_Tc"],
            Tcw_Tc=case_meta["Tcw_Tc"],
            grid=case_meta["grid"],
            iteration=int(f.attrs["Iteration"][0]),
            time=float(f.attrs["Time"][0]),
            averaging_time=float(f.attrs["AveragingTime"][0]),
            extra=extra,
        )

    def _validate(self, metadata: DNSCaseMetadata, fields: dict[str, np.ndarray]) -> None:
        if not fields:
            raise ValueError(f"Case {metadata.case_id}: no fields found in file")
        if self.required_variables:
            missing = set(self.required_variables) - set(fields.keys())
            if missing:
                raise ValueError(f"Case {metadata.case_id}: missing variables {missing}")
        shapes = {v.shape for v in fields.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"Case {metadata.case_id}: inconsistent field shapes: "
                f"{ {k: v.shape for k, v in fields.items()} }"
            )

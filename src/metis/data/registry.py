"""One authoritative representation of a DNS case (milestone R2).

Before this, "what is case01" was spread across `HDF5Reader`
(`_default_metadata_path`), `metis.features.regime` (which rebuilds
`data_root / "processed" / case_id / "metadata.json"` and
`data_root / "processed_slices" / ...` by hand), the dataset config
YAMLs, and the external slice product. `CaseRegistry` centralises it:

    registry = CaseRegistry.from_config()          # resolves the data root
    case = registry["case01"]                       # -> CaseDescriptor
    case.Pb_Pc, case.nx, case.has_slices
    dns_case   = case.load()                        # -> DNSCase
    slice_case = case.load_slice("s3_center")       # -> SliceCase

Analysis code asks the registry for a case; it never reconstructs paths.

Standard layout under the data root (see `metis.config`)::

    <root>/raw/<case_id>/*.h5
    <root>/processed/<case_id>/metadata.json        <- the authoritative schema
    <root>/processed_slices/<case_id>/<slice_id>/
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from metis.config import load_config, resolve_data_root
from metis.data.ingestion.hdf5_reader import DNSCase, HDF5Reader, latest_snapshot
from metis.data.ingestion.slice_reader import SliceCase, SliceReader

# The authoritative case-metadata schema: keys every processed/<case>/
# metadata.json must carry for the registry to describe it. `grid` must in
# turn hold Nx/Ny/Nz. Everything else in the file is kept on `.extra`.
REQUIRED_METADATA_KEYS = ("Pb_Pc", "Thw_Tc", "Tcw_Tc", "grid")
GRID_KEYS = ("Nx", "Ny", "Nz")

DEFAULT_RAW_SUBDIR = "raw"
DEFAULT_PROCESSED_SUBDIR = "processed"
DEFAULT_SLICES_SUBDIR = "processed_slices"


@dataclass(frozen=True)
class CaseDescriptor:
    """Everything needed to locate and load one DNS case. Immutable; built
    by `CaseRegistry`, not constructed directly in analysis code."""

    case_id: str
    Pb_Pc: float
    Thw_Tc: float
    Tcw_Tc: float
    nx: int
    ny: int
    nz: int
    raw_dir: Path
    processed_dir: Path
    slice_root: Path | None = None
    n_snapshots: int | None = None
    extra: Mapping[str, object] = field(default_factory=dict, compare=False)

    # -- derived paths -------------------------------------------------
    @property
    def metadata_path(self) -> Path:
        return self.processed_dir / "metadata.json"

    @property
    def grid(self) -> dict[str, int]:
        return {"Nx": self.nx, "Ny": self.ny, "Nz": self.nz}

    # -- availability of data products -------------------------------
    @property
    def has_processed(self) -> bool:
        return self.metadata_path.exists()

    @property
    def has_raw(self) -> bool:
        return self.raw_dir.is_dir() and any(self.raw_dir.glob("*.h5"))

    @property
    def has_slices(self) -> bool:
        return bool(self.available_slices())

    def available_slices(self) -> list[str]:
        """slice_ids that actually have a readable directory for this case."""
        if self.slice_root is None or not self.slice_root.is_dir():
            return []
        return sorted(
            p.name for p in self.slice_root.iterdir()
            if (p / "metadata.json").exists()
        )

    # -- resolve concrete inputs -----------------------------------
    def raw_snapshot(self) -> Path:
        """Path to the latest raw HDF5 snapshot; raises if none exist."""
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                f"{self.case_id}: no raw directory at {self.raw_dir}"
            )
        return latest_snapshot(self.raw_dir)  # raises FileNotFoundError if empty

    def slice_dir(self, slice_id: str) -> Path:
        if self.slice_root is None:
            raise FileNotFoundError(
                f"{self.case_id}: no slice data (expected under "
                f"processed_slices/{self.case_id}/)"
            )
        path = self.slice_root / slice_id
        if not (path / "metadata.json").exists():
            avail = self.available_slices()
            raise FileNotFoundError(
                f"{self.case_id}: slice {slice_id!r} not found at {path}. "
                f"Available: {avail or 'none'}"
            )
        return path

    # -- load through the standard readers ---------------------------
    def load(self, **reader_kwargs) -> DNSCase:
        """Read the raw snapshot into a `DNSCase` via `HDF5Reader`."""
        return HDF5Reader(**reader_kwargs).read(
            self.raw_snapshot(), metadata_path=self.metadata_path
        )

    def load_slice(self, slice_id: str, **reader_kwargs) -> SliceCase:
        """Read one slice into a `SliceCase` via `SliceReader`."""
        return SliceReader(**reader_kwargs).read(self.slice_dir(slice_id))

    def require(
        self,
        *,
        raw: bool = False,
        slices: Sequence[str] | None = None,
    ) -> CaseDescriptor:
        """Assert the data products an analysis needs are present, with a
        useful error if not. Returns self so it can be chained."""
        if not self.has_processed:
            raise FileNotFoundError(
                f"{self.case_id}: metadata not found at {self.metadata_path}"
            )
        if raw and not self.has_raw:
            raise FileNotFoundError(
                f"{self.case_id}: no raw .h5 snapshot under {self.raw_dir}"
            )
        for slice_id in slices or ():
            self.slice_dir(slice_id)  # raises FileNotFoundError with the list
        return self


def _grid_dims(grid: Mapping[str, object], case_id: str) -> tuple[int, int, int]:
    missing = [k for k in GRID_KEYS if k not in grid]
    if missing:
        raise ValueError(f"{case_id}: metadata 'grid' missing keys {missing}")
    return tuple(int(grid[k]) for k in GRID_KEYS)  # type: ignore[return-value]


class CaseRegistry:
    """Discovers and describes the DNS cases available under one data root.

    Cases are discovered from `<root>/<processed_subdir>/<case_id>/
    metadata.json`; `case_id` is the directory name.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        raw_subdir: str = DEFAULT_RAW_SUBDIR,
        processed_subdir: str = DEFAULT_PROCESSED_SUBDIR,
        slices_subdir: str = DEFAULT_SLICES_SUBDIR,
    ):
        self.data_root = Path(data_root)
        self.raw_root = self.data_root / raw_subdir
        self.processed_root = self.data_root / processed_subdir
        self.slices_root = self.data_root / slices_subdir

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        data_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> CaseRegistry:
        """Build a registry from `metis.config`: data root resolved via
        `resolve_data_root` (CLI value > config `data.root` > env), and
        the `paths:` block for the subdir names."""
        cfg = dict(config) if config is not None else load_config(config_path)
        root = resolve_data_root(cli_value=data_root, config_path=config_path)
        paths = cfg.get("paths") or {}
        return cls(
            root,
            raw_subdir=paths.get("raw", DEFAULT_RAW_SUBDIR),
            processed_subdir=paths.get("processed", DEFAULT_PROCESSED_SUBDIR),
            slices_subdir=paths.get("slices", DEFAULT_SLICES_SUBDIR),
        )

    def case_ids(self) -> list[str]:
        if not self.processed_root.is_dir():
            return []
        return sorted(
            p.name for p in self.processed_root.iterdir()
            if (p / "metadata.json").exists()
        )

    def __contains__(self, case_id: object) -> bool:
        return isinstance(case_id, str) and (
            self.processed_root / case_id / "metadata.json"
        ).exists()

    def __len__(self) -> int:
        return len(self.case_ids())

    def __iter__(self) -> Iterator[CaseDescriptor]:
        return (self[c] for c in self.case_ids())

    def __getitem__(self, case_id: str) -> CaseDescriptor:
        meta_path = self.processed_root / case_id / "metadata.json"
        if not meta_path.exists():
            known = self.case_ids()
            raise KeyError(
                f"unknown case {case_id!r}: no {meta_path}. "
                f"Known cases: {known or 'none'}"
            )
        meta = json.loads(meta_path.read_text())
        missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
        if missing:
            raise ValueError(f"{case_id}: metadata.json missing keys {missing}")
        if meta.get("case") not in (None, case_id):
            raise ValueError(
                f"{case_id}: metadata.json 'case' is {meta['case']!r}, "
                f"expected {case_id!r} (directory name)"
            )
        nx, ny, nz = _grid_dims(meta["grid"], case_id)
        slice_root = self.slices_root / case_id
        return CaseDescriptor(
            case_id=case_id,
            Pb_Pc=float(meta["Pb_Pc"]),
            Thw_Tc=float(meta["Thw_Tc"]),
            Tcw_Tc=float(meta["Tcw_Tc"]),
            nx=nx, ny=ny, nz=nz,
            raw_dir=self.raw_root / case_id,
            processed_dir=self.processed_root / case_id,
            slice_root=slice_root if slice_root.is_dir() else None,
            n_snapshots=meta.get("n_snapshots"),
            extra={
                k: v for k, v in meta.items()
                if k not in (*REQUIRED_METADATA_KEYS, "case", "n_snapshots")
            },
        )

    def get(self, case_id: str, default=None):
        try:
            return self[case_id]
        except KeyError:
            return default

    def descriptors(
        self, case_ids: Sequence[str] | None = None
    ) -> list[CaseDescriptor]:
        ids = list(case_ids) if case_ids is not None else self.case_ids()
        return [self[c] for c in ids]

    def require(
        self, case_id: str, *, raw: bool = False, slices: Sequence[str] | None = None
    ) -> CaseDescriptor:
        return self[case_id].require(raw=raw, slices=slices)

"""Validators for DNS cases, slices, and case collections (milestone R3).

Catch bad or incompatible DNS data *before* scientific algorithms run on
it. Each `validate_*` returns a `ValidationReport`; nothing here raises on
a data problem (call `report.raise_if_failed()` for that).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from metis.data.ingestion.hdf5_reader import DNSCase
from metis.data.ingestion.slice_reader import SliceCase
from metis.data.registry import CaseDescriptor
from metis.data.validation.report import Checks, ValidationReport

# Field base-names (after stripping an "avg_" prefix) that must be
# strictly positive everywhere — density, absolute temperature, pressure,
# transport coefficients, specific heats, speed of sound.
POSITIVE_BASE_NAMES = frozenset(
    {"rho", "T", "P", "mu", "kappa", "c_p", "c_v", "sos"}
)


def _positive_check_name(field_name: str) -> str | None:
    base = field_name.removeprefix("avg_")
    return base if base in POSITIVE_BASE_NAMES else None


# --------------------------------------------------------------------- #
# DNS 3-D snapshot
# --------------------------------------------------------------------- #
def validate_dns_case(case: DNSCase) -> ValidationReport:
    """Structural + numerical checks on a loaded `DNSCase`."""
    c = Checks(case.metadata.case_id)
    fields = case.fields
    grid = case.metadata.grid

    if not c.expect(bool(fields), "fields_present", "no fields in case"):
        return c.report()

    shapes = {v.shape for v in fields.values()}
    if not c.expect(
        len(shapes) == 1, "field_shapes_consistent",
        f"fields have differing shapes: {sorted(map(str, shapes))}",
    ):
        return c.report()
    shape = next(iter(shapes))

    if c.expect(len(shape) == 3, "field_ndim", f"expected 3-D fields, got shape {shape}"):
        # RHEA arrays are [z, y, x] with a ghost layer per side.
        expected = tuple(grid[k] + 2 for k in ("Nz", "Ny", "Nx"))
        c.expect(
            shape == expected, "grid_matches_metadata",
            f"field shape {shape} != metadata grid + ghost cells {expected} "
            f"([z,y,x] of grid={grid})",
        )

    for axis, name in enumerate(("z", "y", "x")):
        coord = case.coordinates.get(name)
        if coord is None:
            c.error("coordinates_present", f"missing {name} coordinate")
            continue
        if len(shape) == 3 and coord.shape[0] != shape[axis]:
            c.error(
                "coordinate_length",
                f"{name} coordinate has {coord.shape[0]} points, "
                f"field axis {axis} has {shape[axis]}",
            )
        interior = coord[1:-1] if coord.size > 2 else coord
        if not np.all(np.isfinite(coord)):
            c.error("coordinate_finite", f"{name} coordinate has non-finite values")
        elif interior.size and np.ptp(interior) == 0:
            # A constant coordinate line usually means the reader picked
            # the wrong mesh axis, not corrupt data — warn, don't fail.
            c.warn("coordinate_degenerate",
                   f"{name} coordinate is constant ({interior.flat[0]:.4g}); check the mesh axis")
        # Only the interior need increase monotonically: ghost cells on a
        # periodic direction (x, z in RHEA) legitimately wrap around.
        elif not np.all(np.diff(interior) > 0):
            c.error("coordinate_monotonic",
                    f"{name} interior coordinate is not strictly increasing")

    for fname, arr in fields.items():
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan:
            c.error("no_nan", f"{fname}: {n_nan} NaN value(s)")
        if n_inf:
            c.error("no_inf", f"{fname}: {n_inf} infinite value(s)")
        if n_nan or n_inf:
            continue

        base = _positive_check_name(fname)
        if base is not None and arr.min() <= 0:
            c.error(
                "physically_positive",
                f"{fname}: min {arr.min():.4g} <= 0 (expected strictly positive)",
            )
        if fname.startswith("rmsf_") and arr.min() < 0:
            c.error(
                "rms_nonnegative",
                f"{fname}: min {arr.min():.4g} < 0 (RMS fluctuation must be >= 0)",
            )
        if arr.size > 1 and not fname.startswith("tag_") and np.ptp(arr) == 0:
            c.warn("field_not_constant", f"{fname}: constant value {arr.flat[0]:.4g}")

    return c.report()


# --------------------------------------------------------------------- #
# Homogeneous-plane slice
# --------------------------------------------------------------------- #
def validate_slice_case(case: SliceCase) -> ValidationReport:
    """Structural + numerical checks on a loaded `SliceCase`."""
    subject = f"{case.metadata.case_id}/{case.metadata.slice_id}"
    c = Checks(subject)
    snaps = case.snapshots
    grid = case.metadata.grid  # {"NX": .., "NZ": ..}

    if not c.expect(bool(snaps), "snapshots_present", "no snapshot fields"):
        return c.report()

    shapes = {v.shape for v in snaps.values()}
    if not c.expect(
        len(shapes) == 1, "snapshot_shapes_consistent",
        f"snapshot fields have differing shapes: {sorted(map(str, shapes))}",
    ):
        return c.report()
    shape = next(iter(shapes))

    if c.expect(len(shape) == 3, "snapshot_ndim", f"expected (n, NX, NZ), got {shape}"):
        nx, nz = grid.get("NX"), grid.get("NZ")
        if nx is not None and nz is not None:
            c.expect(
                shape[1:] == (nx, nz), "slice_grid_matches_metadata",
                f"snapshot plane {shape[1:]} != metadata (NX, NZ) = {(nx, nz)}",
            )
        expected_n = case.metadata.extra.get("n_snapshots")
        if expected_n is not None and shape[0] != expected_n:
            c.warn(
                "n_snapshots_matches_metadata",
                f"{shape[0]} snapshots on disk, metadata says {expected_n}",
            )

    for name in ("x", "z"):
        coord = case.coordinates.get(name)
        if coord is None:
            c.error("coordinates_present", f"missing {name} coordinate")
        elif not np.all(np.isfinite(coord)):
            c.error("coordinate_finite", f"{name} coordinate has non-finite values")
        elif not np.all(np.diff(coord) > 0):
            c.error("coordinate_monotonic", f"{name} coordinate not strictly increasing")

    for fname, arr in snaps.items():
        if np.isnan(arr).any():
            c.error("no_nan", f"snapshots_{fname}: contains NaN")
        if np.isinf(arr).any():
            c.error("no_inf", f"snapshots_{fname}: contains inf")
        mean_field = case.mean.get(fname)
        if mean_field is not None and mean_field.shape != shape[1:]:
            c.error(
                "mean_shape",
                f"mean_{fname} shape {mean_field.shape} != snapshot plane {shape[1:]}",
            )
        # snapshots are stored as fluctuations: their time-mean should be
        # small next to the mean field (slice_reader docstring: ~0.03).
        if mean_field is not None and np.isfinite(arr).all():
            scale = float(np.abs(mean_field).mean()) or 1.0
            drift = float(np.abs(arr.mean()))
            if drift > 0.1 * scale:
                c.warn(
                    "snapshots_are_fluctuations",
                    f"{fname}: |time-mean of snapshots| {drift:.3g} is large "
                    f"vs mean-field scale {scale:.3g}",
                )
        # duplicate frames (hash each frame's bytes)
        if arr.ndim == 3:
            seen: set[bytes] = set()
            dups = 0
            for frame in arr:
                h = frame.tobytes()
                if h in seen:
                    dups += 1
                seen.add(h)
            if dups:
                c.warn("no_duplicate_snapshots", f"{fname}: {dups} duplicate frame(s)")

    return c.report()


# --------------------------------------------------------------------- #
# Descriptor (files on disk, before loading)
# --------------------------------------------------------------------- #
def validate_case(descriptor: CaseDescriptor, *, load: bool = True) -> ValidationReport:
    """Validate a `CaseDescriptor`: metadata sanity and data-product
    presence, then (if `load`) read the raw snapshot and run
    `validate_dns_case` on it."""
    c = Checks(descriptor.case_id)

    c.expect(descriptor.has_processed,
             "processed_present", f"no metadata at {descriptor.metadata_path}")
    for dim, n in (("nx", descriptor.nx), ("ny", descriptor.ny), ("nz", descriptor.nz)):
        c.expect(isinstance(n, int) and n > 0, "grid_positive", f"{dim} = {n!r}")

    ts = descriptor.extra.get("timesteps")
    if isinstance(ts, Sequence) and not isinstance(ts, str):
        c.expect(list(ts) == sorted(ts), "timesteps_ordered",
                 "metadata 'timesteps' is not ascending")
        done = descriptor.extra.get("timesteps_done")
        if isinstance(done, Sequence) and not isinstance(done, str):
            missing = set(done) - set(ts)
            c.expect(not missing, "timesteps_done_subset",
                     f"'timesteps_done' has {len(missing)} value(s) not in 'timesteps'")

    report = c.report()

    if load and descriptor.has_raw:
        try:
            dns_case = descriptor.load()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the validator
            c2 = Checks(descriptor.case_id)
            c2.error("loadable", f"could not load raw snapshot: {exc}")
            report.extend(c2.report())
        else:
            report.extend(validate_dns_case(dns_case))
    elif load and not descriptor.has_raw:
        c2 = Checks(descriptor.case_id)
        c2.warn("raw_present", f"no raw .h5 under {descriptor.raw_dir}; skipped numerical checks")
        report.extend(c2.report())

    return report


# --------------------------------------------------------------------- #
# Cross-case compatibility (before combining cases)
# --------------------------------------------------------------------- #
def validate_compatibility(
    cases: Sequence[CaseDescriptor | DNSCase],
) -> ValidationReport:
    """Check a set of cases can be combined: same grid, and (for loaded
    `DNSCase`s) the same set of field names."""
    c = Checks(f"compatibility[{len(cases)} cases]")
    if not c.expect(len(cases) >= 2, "enough_cases", "need >= 2 cases to compare"):
        return c.report()

    grids: dict[str, tuple[int, int, int]] = {}
    field_sets: dict[str, frozenset[str]] = {}
    for case in cases:
        if isinstance(case, CaseDescriptor):
            cid = case.case_id
            grids[cid] = (case.nx, case.ny, case.nz)
        else:  # DNSCase
            cid = case.metadata.case_id
            g = case.metadata.grid
            grids[cid] = (g["Nx"], g["Ny"], g["Nz"])
            field_sets[cid] = frozenset(case.fields)

    distinct_grids = set(grids.values())
    c.expect(
        len(distinct_grids) == 1, "grids_match",
        f"cases have {len(distinct_grids)} distinct grids: "
        + ", ".join(f"{k}={v}" for k, v in grids.items()),
    )
    if field_sets:
        distinct_fields = set(field_sets.values())
        if len(distinct_fields) != 1:
            ref_cid, ref = next(iter(field_sets.items()))
            for cid, fs in field_sets.items():
                diff = ref ^ fs
                if diff:
                    c.error(
                        "field_names_match",
                        f"{cid} field set differs from {ref_cid} by {sorted(diff)}",
                    )

    return c.report()

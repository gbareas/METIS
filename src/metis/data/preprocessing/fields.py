"""Stateless field-array preprocessing helpers (milestone R4).

These need no `fit` (nothing is learned from the data), so they are plain
functions rather than `Transform` subclasses: ghost-cell removal,
variable selection, spatial subsampling.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from metis.data.ingestion.hdf5_reader import DNSCase


def strip_ghost_cells(arr: np.ndarray, n: int = 1) -> np.ndarray:
    """Drop `n` cells from both ends of every axis (RHEA's ghost layer)."""
    if n <= 0:
        return arr
    return arr[tuple(slice(n, -n) for _ in range(arr.ndim))]


def select_variables(
    fields: Mapping[str, np.ndarray], names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return a new dict with just `names`, in that order. Raises
    `KeyError` listing everything missing."""
    missing = [n for n in names if n not in fields]
    if missing:
        raise KeyError(f"fields not present: {missing} (have: {sorted(fields)})")
    return {n: fields[n] for n in names}


def subsample(
    arr: np.ndarray, factor: int, axes: Iterable[int] | None = None
) -> np.ndarray:
    """Take every `factor`-th sample along `axes` (all axes by default)."""
    if factor <= 1:
        return arr
    axes = range(arr.ndim) if axes is None else axes
    slices = [slice(None)] * arr.ndim
    for ax in axes:
        slices[ax] = slice(None, None, factor)
    return arr[tuple(slices)]


def interior_fields(case: DNSCase, n: int = 1) -> dict[str, np.ndarray]:
    """`case.fields` with the ghost layer stripped from every field."""
    return {name: strip_ghost_cells(arr, n) for name, arr in case.fields.items()}

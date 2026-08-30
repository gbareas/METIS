"""Preprocessing layer (milestone R4).

    from metis.data.preprocessing import StandardScaler, save, load

Transforms are `fit` on training data only, then `transform` applied to
every split; fitted transforms serialise with `save` / `load`. Stateless
field helpers (`strip_ghost_cells`, `select_variables`, `subsample`,
`interior_fields`) are plain functions.
"""
from metis.data.preprocessing.base import (
    NotFittedError,
    Transform,
    from_dict,
    load,
    save,
    to_dict,
)
from metis.data.preprocessing.fields import (
    interior_fields,
    select_variables,
    strip_ghost_cells,
    subsample,
)
from metis.data.preprocessing.scalers import MinMaxScaler, StandardScaler

__all__ = [
    "MinMaxScaler",
    "NotFittedError",
    "StandardScaler",
    "Transform",
    "from_dict",
    "interior_fields",
    "load",
    "save",
    "select_variables",
    "strip_ghost_cells",
    "subsample",
    "to_dict",
]

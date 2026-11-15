# DNS data layout

METIS expects one **data root** holding three products, one sub-directory
per case (`case01`, `case02`, …):

```
<data_root>/
  raw/<case>/*.h5                     one RHEA snapshot per file (an iteration
                                      checkpoint); a case dir holds several
  processed/<case>/metadata.json      case-level physics metadata (authoritative)
  processed_slices/<case>/<slice_id>/ homogeneous-plane slice product
      snapshots_<field>.npy           (n_snapshots, NX, NZ) fluctuation fields
      mean_<field>.npy                (NX, NZ) time-mean field
      grid.npz                        x (NX,), z (NZ,), y_loc  [metres]
      metadata.json                   case + slice descriptors
```

Point METIS at `<data_root>` with `--data-root`, `$METIS_DATA_ROOT`, or
`data.root:` in `configs/default.yaml` (resolution order: flag > config >
env). Sub-directory names are configurable under `paths:` but default to
`raw` / `processed` / `processed_slices`.

## `processed/<case>/metadata.json`

The authoritative case schema. Required keys (`REQUIRED_METADATA_KEYS` in
`metis.data.registry`): **`Pb_Pc`**, **`Thw_Tc`**, **`Tcw_Tc`**,
**`grid`** (with `Nx`, `Ny`, `Nz`). Everything else — `case`,
`case_number`, `n_snapshots`, `timesteps`, `fields`, `output_shape`,
`dtype`, … — is carried on `CaseDescriptor.extra`.

`Pb_Pc` / `Thw_Tc` / `Tcw_Tc` are the bulk-pressure and hot/cold-wall
temperature ratios (over the CO₂ critical point) the whole case family is
parametrised by.

## RHEA array convention: `[z, y, x]`

RHEA writes every root-level dataset in a raw HDF5 file with shape
**`(Nz+2, Ny+2, Nx+2)`** — `z` along axis 0, `y` along axis 1, `x` along
axis 2 — with **one ghost cell per side**. Ghost cells are wall-mirror
cells on the wall-normal (`y`) axis and wrap around on the periodic
(`x`, `z`) axes. `HDF5Reader` returns per-axis 1-D `coordinates` and a
`fields` dict; `wall_normal_profiles` averages the two homogeneous axes
(0 and 2) and keeps `y` (axis 1).

`metis.testing.mock_dns` produces the same `[z, y, x]` layout.

## Slices

Only **XZ homogeneous-plane** slices (`slice_prefix` starting
`plane_XZ_`) are read; `x` and `z` are the periodic directions that
spectra and POD need. Wall-normal (XY/ZY) slices use a different schema
and are out of scope. Slice snapshots are stored as fluctuations (the
time-mean is subtracted upstream); no physical timestep is stored, only
solver iteration numbers, so temporal/frequency analysis is not
available from this product.

## Using it

```python
from metis.data.registry import CaseRegistry

reg = CaseRegistry.from_config()          # resolves the data root
case = reg["case01"]                      # -> CaseDescriptor
case.Pb_Pc, case.nx, case.has_slices, case.available_slices()
dns   = case.load()                       # -> DNSCase
sl    = case.load_slice("s3_center")      # -> SliceCase
case.require(raw=True, slices=["s3_center"])   # explicit, useful errors
```

The 11 simulated cases are `case01`–`case09` (a 3×3 `Pb_Pc × Thw_Tc`
training grid) plus `case10` and `case15` (out-of-distribution); see
`research_protocol.md`.

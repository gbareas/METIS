# Virtual-sensor temporal-ML protocol (Phase C)

**Status: C1 FROZEN 2026-08-30.** Machine-readable parameters live in
`configs/experiments/virtual_sensors_v1.yaml`; neither may change without
a `FINDINGS.md` entry. Per `METIS_detailed_next_steps_updated.md`
§30–36, §53.

Phase C turns the DNS slice product into a conventional industrial
multivariate time-series problem:

```
DNS XZ slice snapshots  →  physically-placed virtual probes
                        →  plane-aggregated sensor channels
                        →  windowed dataset  →  forecasting (+ regime detection)
```

C1 is **sensor / variable selection only** — the dataset builder (C2),
splits (C4), and models (C5–C8) are later milestones.

---

## 1. Data product

`data/processed_slices/<case>/<slice_id>/` — XZ homogeneous-plane slices
(`metis.data.ingestion.slice_reader.SliceReader` / `CaseRegistry`), the
same product I2-B used. Per case/slice: **96 × 96** grid, **500
fluctuation snapshots** (time-mean already subtracted upstream), fields
`u`, `T`, `cp`, plus `mean_<field>.npy` and `grid.npz` (`x`, `z`,
`y_loc`).

**Sampling.** `metadata["timesteps"]` is uniform at **2500 solver
iterations** between consecutive slice snapshots for every case — a
clean, evenly spaced discrete series. There is **no authoritative
physical Δt** for this product (only solver iteration count), so
per §35 this is **sample-step forecasting**: horizons are reported in
snapshot steps (1 step = 2500 iterations), never in seconds or Hz.

**Cases.** case01–09 span the 3×3 operating grid (`Pb_Pc` ∈ {1.5, 2.0,
5.0} × `Thw_Tc` ∈ {1.1, 1.2, 1.4}); case10 and case15 are off-grid
thermal forcing at `Pb_Pc = 1.5`. All 5 probe slices below exist for all
11 cases.

---

## 2. Virtual probes (§31)

Five physically-motivated wall-normal stations, each an existing XZ slice
product — **not** an arbitrary point cloud:

| probe id | slice_id | region | why this plane |
|---|---|---|---|
| `cw_buffer` | `s5_y_plus_10_cw` | cold-wall buffer layer, y⁺ ≈ 10 | peak turbulence production on the dense (cold-wall) side |
| `centre` | `s3_center` | channel centre | bulk turbulence, minimal wall / pseudo-boiling interference (the I2-B slice) |
| `hw_buffer` | `s8_y_plus_10_hw` | hot-wall buffer layer, y⁺ ≈ 10 | near-wall turbulence on the light (hot-wall) side; the cw/hw pair measures transcritical asymmetry |
| `pseudoboiling` | `s1_max_cp_f` | plane of maximum `cp` (Widom-line crossing) | the transcritical-specific structure — largest thermodynamic sensitivity, the natural regime signal |
| `max_u` | `s2_max_u` | plane of maximum mean streamwise velocity | its own wall-normal location encodes the density-gradient-induced velocity-profile skew |

`grid.npz["y_loc"]` (plane position, metres) is recorded per (case,
probe) as static metadata.

---

## 3. Sensor channels (§31 — meaningful, not hundreds)

Each 96 × 96 plane is reduced to scalars per snapshot by aggregating over
the two homogeneous directions. Per **probe × field (`u`, `T`, `cp`)**,
two aggregates — both temporally coherent and physically readable:

| channel suffix | definition | reads as |
|---|---|---|
| `_pmf` | plane-mean fluctuation ⟨f′⟩ₓ_z(t) | instantaneous large-scale departure (≈ zero-mean, low-frequency) |
| `_rms` | plane RMS fluctuation √⟨f′²⟩ₓ_z(t) | turbulence intensity (strictly positive, second-order) |

→ **5 probes × 3 fields × 2 aggregates = 30 sensor channels**, 500
sample-steps per case. Channel id: `<probe>.<field>_<suffix>` (e.g.
`pseudoboiling.T_rms`).

**Static exogenous context** (constant in time, per case): `Pb_Pc`,
`Thw_Tc`, `Tcw_Tc`, and per-probe `y_loc`. Covariates for the
forecaster, never targets.

### Rationale for the aggregation choice

A fluctuation plane has ⟨f′⟩ ≈ 0 by construction, so a single arbitrary
pixel would be a noisy, location-dependent probe with no canonical
choice. `_pmf` keeps the honest low-frequency bulk signal; `_rms` keeps
the second-order intensity. The per-case time-mean plane value
(`mean_<field>` spatial mean) is recorded as static metadata so an
absolute level can be reconstructed if a model needs it.

### Optional derived observable (secondary, not frozen into v1 targets)

`<probe>.u_peakk` — premultiplied-spectrum peak wavenumber along x per
snapshot — links the sensor stream back to the regime-discovery work
(FINDINGS §1–5). Built only if C10 (regime-aware evaluation) needs it.

---

## 4. Task formulation (C2+, frozen here for the builder)

**Forecasting (task 1, §32).** Multivariate, all 30 channels:

```
X_{t-L+1 : t}   →   X_{t+1 : t+H}
```

- context length `L = 32` steps;
- horizons `H ∈ {1, 4, 8, 16}` steps (horizon-degradation analysis);
- windows are stride-1 within a case, never spanning cases.

**Regime / anomaly detection (task 2, §33).** Secondary: can a short
window of sensor channels classify the 3×3 regime or flag OOD? Deferred
past the first validated forecaster; noted so C2's dataset keeps the
per-window case / operating-point labels needed for it.

---

## 5. Splits (C4 — chronological, physical generalisation; §34)

Same convention as `regime-v1` / I2-B:

- **train** — case01–09, steps 0–399 (first 80 %);
- **val** — case01–09, steps 400–499 (chronological hold-out, later
  solver iterations, no shuffling);
- **OOD** — all 500 steps of case10 and case15 (unseen off-grid thermal
  forcing at fixed pressure).
- **secondary (robustness only)** — pressure hold-out: train case01–06,
  OOD case07–09 (`Pb_Pc = 5.0`). Reported alongside, not part of the
  headline decision unless it disagrees sharply.

Standardisation: per-channel mean/std fit on the **train** split only,
applied unchanged to val and OOD. Random splits are never used as
evidence for a forecasting claim.

---

## 6. Baselines & models (C5–C8, for reference)

1. **persistence** — X̂_{t+h} = X_t (mandatory reference);
2. **autoregressive / lagged linear regression** — per-channel and VAR;
3. **gradient-boosted trees** (XGBoost / LightGBM) on lag + rolling
   statistical features;
4. **LSTM** (then a temporal CNN only if it earns its place).

Transformer-style models only if the simpler ones show a specific,
named limitation.

---

## 7. Evaluation (C9–C10)

- per-horizon `relative_l2` / `rmse` / `mae` / `r2` vs persistence, on
  val and OOD, per channel and aggregated;
- skill score `1 − MSE_model / MSE_persistence`;
- OOD degradation ratio (OOD error / val error);
- regime-aware slice: does forecast skill hold across `Pb_Pc` /
  `Thw_Tc` bands, and does it collapse on the OOD thermal forcing?

A model is "better" only under the §17 rule: an ML metric improves **and**
no physical/regime diagnostic regresses.

---

## 8. Deliverables / order (§53)

1. **C1** — this document + `configs/experiments/virtual_sensors_v1.yaml`
   (sensor & variable selection). ← *done*
2. **C2** — `build_virtual_sensor_dataset(registry, config)` →
   per-split channel matrices + window index + `metadata.json`, under
   `artifacts/datasets/virtual_sensors_v1/`.
3. **C3** — Parquet (long-form) persistence of the channel table.
4. **C4** — leakage-safe chronological windowing (in C2's builder;
   verified by a test that no window crosses the train/val boundary or a
   case boundary).
5. **C5–C8** — persistence → AR/linear → GBT → LSTM.
6. **C9** — OOD operating-condition tests.
7. **C10** — regime-aware evaluation.
8. **C11** — MLflow tracking + register the validated forecaster.

**Exit criterion:** METIS contains a credible end-to-end sample-step
time-series forecasting workflow.

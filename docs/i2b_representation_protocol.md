# I2-B — slice-level representation-learning protocol

**Status: FROZEN 2026-08-30 (B1).** The machine-readable parameters live
in `configs/experiments/i2b_representation_v1.yaml`; neither may change
without a `FINDINGS.md` entry. Per
`METIS_detailed_next_steps_updated.md` §24–29, §52, §57 — the model is
**not** tuned before the study runs to its decision gate (B7).

Freeze decisions (confirmed): slice `s3_center` / field `u`; primary OOD
= case10 + case15 (series convention) with a secondary Pb_Pc = 5.0
pressure-holdout; `Trainer` gains a mini-batch loop in B4 (batch 64); 5
seeds and the `k ∈ {2,4,8,16,32}` grid.

Supersedes nothing; extends the Track A/B reference work (`research_protocol.md`,
`FINDINGS.md` §1–6). I2-A (case-level, N = 9 training cases) reached a
negative result (`FINDINGS.md` §6): with 9 samples a nonlinear
autoencoder does not robustly beat PCA and is initialisation-sensitive.
I2-B asks the same scientific question in a setting where N is large
enough for the question to be fair.

---

## 1. Scientific question

> Does a nonlinear representation learned from **2-D DNS slice fields**
> capture physically meaningful transcritical-flow structure that a
> linear PCA / POD basis does not represent as efficiently or as
> robustly — in particular across changing thermodynamic operating
> conditions?

### Falsifiable hypothesis (H-I2B)

A convolutional autoencoder trained on centre-plane streamwise-velocity
fluctuation snapshots produces, at a **matched latent dimension**, a
representation that is at least one of:

- **more compressive** — lower reconstruction relative-L2 at equal latent
  dim, or equal error at fewer latent coordinates;
- **more physically faithful** — reconstructed fields preserve the
  wall-normal RMS profile and the wavenumber spectrum better than the PCA
  reconstruction;
- **more OOD-robust** — smaller degradation from validation to held-out
  operating conditions;
- **more physically interpretable** — latent coordinates correlate more
  strongly with `Pb_Pc` / `Thw_Tc` / `Tcw_Tc` (or per-snapshot derived
  quantities) than the leading PCA scores.

### Falsification / stop condition

If, across independent seeds, the autoencoder does **none** of the above
by more than the seed-to-seed noise — or improves a generic ML metric
while regressing a physical diagnostic (`assess_model` → `is_better =
False`) — then linear representations are sufficient for this flow
structure under these conditions. **Record the negative result in
`FINDINGS.md` §7 and stop** (do not escalate to VAE / U-Net / FNO).

Both outcomes are acceptable. The platform must support stopping a method
when the evidence says it is unnecessary.

---

## 2. Data

### Product

`data/processed_slices/<case>/<slice_id>/` — XZ homogeneous-plane slices,
read by `metis.data.ingestion.slice_reader.SliceReader` /
`CaseRegistry`. Each case/slice: **96 × 96** grid, **500 fluctuation
snapshots** (time-mean already subtracted upstream), fields `u`, `T`,
`cp`.

### Frozen choices (start with one field, one slice — §26)

| choice | value | why |
|---|---|---|
| slice | `s3_center` | channel centre: statistically homogeneous, away from wall/pseudo-boiling complications; the slice I2-A's compact POD feature used |
| field | `u` (streamwise velocity fluctuation) | cleanest turbulence structure, most-studied; matches the `bulk`/`rms` focus of Track A/B |
| sample unit | one 2-D snapshot `(96, 96)` | ~500 per case → thousands of training samples, unlike I2-A's 9 |

`T` on the same slice is an optional follow-up (**I2-B-2**), run only if
I2-B-1 is inconclusive rather than clearly negative. Near-wall slices
(`s5_y_plus_10_cw`, `s8_y_plus_10_hw`) and multi-field inputs are out of
scope for this protocol.

### Standardisation

A single global scale per field, **fit on the training split only**
(`metis.data.preprocessing.StandardScaler` over all training snapshots,
flattened): subtract the residual mean (≈ 0 for a fluctuation field),
divide by the global std. Applied unchanged to validation and OOD. No
per-snapshot or per-pixel normalisation.

---

## 3. Splits (§27 — physical generalisation, not random shuffling)

Cases (from `data/processed/<case>/metadata.json`):

```
training grid (3×3 in Pb_Pc × Thw_Tc):
  Pb_Pc \ Thw_Tc   1.1        1.2        1.4
  1.5              case01     case02     case03
  2.0              case04     case05     case06
  5.0              case07     case08     case09
off-grid (Pb_Pc = 1.5, off-grid thermal forcing):
  case10 (Thw_Tc 1.185 / Tcw_Tc 1.035)   case15 (Thw_Tc 1.132 / Tcw_Tc 0.982)
```

### Primary split (`i2b_representation_v1`)

- **Train**: snapshots 0–399 (first 80 %) of every case01–case09.
- **Validation**: snapshots 400–499 (last 20 %) of every case01–case09 —
  a **chronological** hold-out (later solver iterations), no shuffling, so
  it measures generalisation to unseen time, not memorisation.
- **OOD test**: **all 500** snapshots of **case10 and case15** — entire
  operating conditions excluded from training (off-grid thermal forcing
  at fixed pressure).

Train ≈ 9 × 400 = **3 600** samples; val ≈ 900; OOD = 1 000.

### Secondary split (`i2b_pressure_holdout_v1`) — robustness only

- **Train**: case01–case06 (Pb_Pc ∈ {1.5, 2.0}), snapshots 0–399.
- **Validation**: case01–case06, snapshots 400–499.
- **OOD test**: case07, case08, case09 (Pb_Pc = 5.0) — an unseen
  pressure. Reported alongside the primary split; not part of the
  headline accept/stop decision unless it disagrees sharply with it.

Random splits are not used as evidence.

---

## 4. Models

All operate on standardised `(96, 96)` snapshots; all compared at the
same latent dimension `k`.

### Baseline — linear (mandatory)

- **PCA** of the flattened training snapshots `(n_train, 9216)`
  (`metis.models.PCARepresentation`). For a fluctuation snapshot ensemble
  this **is** snapshot POD; cross-checked against `metis.features.pod`
  on the same data (leading singular values must match to float32).
- Report at `k ∈ {2, 4, 8, 16, 32}`.

### Candidate — convolutional autoencoder (fixed architecture)

A **small** symmetric conv-AE, frozen for the protocol (no architecture
search):

```
encoder:  Conv(1→16, 3×3, stride 2) + act   → 48×48×16
          Conv(16→32, 3×3, stride 2) + act  → 24×24×32
          Conv(32→64, 3×3, stride 2) + act  → 12×12×64
          flatten → Linear(9216 → k)
decoder:  Linear(k → 9216) → reshape 12×12×64
          ConvTranspose mirror ×3 → 96×96×1
activation: GELU;  loss: MSE on the standardised field
```

- Trained via `metis.training.Trainer` (Adam, early stop on validation
  MSE, best-state restore, checkpoint), MLflow-logged via
  `metis.tracking`. Mini-batches (batch 64) — `Trainer` currently does
  full-batch; a mini-batch loop is the one small `Trainer` extension this
  study requires, added before B4 and covered by a test.
- **Latent dim**: same `k` grid as PCA. Headline comparison at the `k`
  where PCA first reaches ≥ 90 % reconstructed variance on the training
  split (determined in B3, then frozen into the config — expected 8 or
  16).
- Do **not** add: VAE, U-Net skip connections, attention, FNO/WNO,
  perceptual/spectral losses. Those are only justified if a positive
  result at this architecture motivates them.

---

## 5. Evaluation

Every metric is reported as **mean ± std over 5 genuinely independent
seeds** (seeds `0..4`; the `FINDINGS.md` §6 / updated-plan §18 seed fix
must be in place — verified by
`tests/unit/test_autoencoder.py::test_different_seeds_give_different_initial_parameters`).

### 5.1 Reconstruction (generic)

On validation and OOD, standardised space:

- `relative_l2`, `rmse`, `mae`, `r2` (`metis.evaluation.metrics`);
- reconstructed variance fraction;
- compression ratio = 9216 / `k`.

### 5.2 Robustness

- seed-to-seed std of every 5.1 metric;
- `latent_stability` (Procrustes-aligned relative-L2 of the latent codes
  across seeds, on a fixed validation subset) —
  `metis.evaluation.representation.latent_stability`;
- **OOD degradation ratio** = (OOD `relative_l2`) / (val `relative_l2`),
  per model; lower is better.

### 5.3 Physical fidelity (`metis.evaluation.physical`)

True vs reconstructed fields, on validation and OOD, per model:

- `profile_agreement` — relative-L2 of the wall-normal (spanwise-averaged
  along one axis; here the plane has no wall-normal axis, so use the
  along-`x` mean and RMS profiles) mean and RMS profiles;
- `spectrum_agreement` — wavenumber-spectrum relative-L2, log-spectrum
  correlation, premultiplied-peak wavenumber shift (along the periodic
  `x` axis, then `z`);
- `pod_energy_agreement` — L1 distance between the leading-`k`
  POD-energy-fraction spectra of the true and reconstructed ensembles.

### 5.4 Latent interpretation

- `latent_physical_correlation` (`metis.evaluation.representation`) of
  each latent axis against `Pb_Pc`, `Thw_Tc`, `Tcw_Tc` (constant within a
  case) and against per-snapshot derived scalars: instantaneous field
  RMS, spectral-peak wavenumber. Compare the max |corr| for the AE latent
  vs the leading PCA scores at the same `k`.

### 5.5 The comparison

For each seed, `metis.evaluation.assessment.assess_model(candidate = AE,
baseline = PCA)` with:

- `ml_metrics` = {`relative_l2` (val), `relative_l2` (OOD),
  `ood_degradation_ratio`, `compression_at_equal_error`};
- `physical_metrics` = {`mean_profile_rel_l2`, `rms_profile_rel_l2`,
  `spectrum_rel_l2`, `spectrum_log_corr`, `energy_fraction_l1`} on both
  val and OOD.

`is_better` requires an ML improvement **and** no physical regression
(beyond `margin`).

---

## 6. Acceptance criteria (§29)

### Positive result — continue

At the headline `k`, **≥ 3 of 5 seeds** give `assess_model.is_better =
True`, i.e. the AE improves reconstruction and/or OOD robustness with no
physical-diagnostic regression; **and** the improvement exceeds the
seed-to-seed std. Then: document it, and I2-B-2 (`T`) / near-wall slices
become justified follow-ups.

### Valuable negative result — stop

Any of:

- no seed reaches `is_better = True`;
- improvements are within seed noise;
- generic metric improves but a physical diagnostic regresses;
- `latent_stability` is poor (AE latent varies materially across seeds
  while PCA is exact).

Then linear representations are sufficient for centre-plane `u`-fluctuation
structure across the tested thermodynamic conditions. Record in
`FINDINGS.md` §7 and stop — no VAE / U-Net / operator escalation off the
back of a negative.

Either way, the result and every metric table are written to
`FINDINGS.md` §7 and `results/i2b_representation.json`, and the trained
best model (if positive) is registered:
`metis registry add i2b_ae_<field>_<slice>_v1 --kind model --run-id <mlflow> --dataset-id <slice-dataset> --scope "..."`.

---

## 7. Compute

Local workstation. Torch with CUDA is available; the conv-AE (3 600
samples, `k ≤ 32`, ~1 M parameters) trains in minutes on GPU, ~10–20 min
on CPU per seed. 5 seeds × 5 latent dims × 1 field ≈ a few GPU-hours
total. No HPC.

---

## 8. Deliverables / execution order (updated-plan §52)

1. **B1** — freeze this document + `configs/experiments/i2b_representation_v1.yaml`.
2. **B2** — Level-2 slice-snapshot dataset builder
   (`metis.data.datasets`): `build_slice_dataset(registry, field,
   slice_id, split_spec) → {train,val,ood}.npz + metadata.json` with a
   train-only scaler and a fingerprint (reuse the §20 source-manifest
   idea). Artifact under `artifacts/datasets/i2b_slice_u_s3_center_v1/`.
3. **B3** — PCA / POD baseline at the `k` grid; pick + freeze the
   headline `k`.
4. **B4** — mini-batch support in `Trainer` (+ test); implement the
   conv-AE (`metis.models`, torch-gated).
5. **B5** — train all seeds × `k`, MLflow-logged.
6. **B6** — 5.1–5.5 evaluation, both splits.
7. **B7** — `assess_model` comparison, accept/stop decision.
8. **B8** — physical-statistic / modal comparison writeup.
9. **B9** — `FINDINGS.md` §7; register the model if positive; a
   `metis report` generator for the study.

**Decision gate after B7**: continue nonlinear representation learning
only if the evidence justifies it.

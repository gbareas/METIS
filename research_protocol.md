# Research Protocol

Status: **Phase 0 complete** (frozen 2026-08-29). Evaluation metrics
revised 2026-08-29 after running the reference task end-to-end — see
`FINDINGS.md` §1-4 and the "Evaluation metrics" section below.

This is the Track A/B reference task (roadmap v2 §40). It governs the first
"does the platform reproduce known physics" milestone; it does not govern
Track C/D (the personal router-agent project), which wraps
`pub5_neural_operators` directly and has its own protocol in
`docs/agent_implementation_plan.md`.

The follow-on **slice-level representation-learning study (I2-B)** has its
own frozen protocol in `docs/i2b_representation_protocol.md` (+ machine-
readable `configs/experiments/i2b_representation_v1.yaml`).

## Primary task

Cross-case regime discovery from raw DNS-derived physical features,
benchmarked against known thermodynamic conditions and existing POD/SPOD
knowledge (roadmap v2 §40). The question: do unsupervised methods applied to
standardized physical feature vectors recover the known `Pb_Pc`/`Thw_Tc`
operating-condition grid without being told it, and which features/variables
drive the separation (roadmap v2 §3, "Physical-regime identification")?

## Hypotheses

- H1: A feature vector built from mean/RMS wall-normal profiles and bulk
  dimensionless groups (`Re_b`, `Pr_b`, `Ec_b`, `Br_b`, `Ma_b` — as already
  computed in `pub4_grassmann_rom/scripts/compute_case_setup_table.py`) is
  sufficient for unsupervised clustering to recover the pressure axis
  (`Pb_Pc`) cleanly, since pressure dominates thermophysical-property
  variation near the pseudo-critical point.
- H2: The wall-temperature axis (`Thw_Tc`/`Tcw_Tc`) is harder to separate
  than pressure from bulk features alone and requires POD-mode-level or
  spectral features (not just bulk numbers) to resolve — i.e. regime
  separability is feature-set-dependent, not just data-dependent.
- H3: case10/case15 (held out, see below) land near the case01-03 cluster in
  feature space (same `Pb_Pc` = 1.5, intermediate `Thw_Tc`) rather than
  forming a distinct regime — thermal-condition interpolation, not regime
  shift, at fixed pressure.

## Cases selected

Reusing the Pub 4/5 series convention (`data/processed/case{NN}/metadata.json`
is authoritative for exact values; wall temperatures for case10/case15 are
the DNS-*measured* values, corrected 2026-06-10 — see
`pub4_grassmann_rom/results/case_setup_table.json` — not any earlier nominal
design values).

**Training grid — case01–case09** (3×3 in `Pb_Pc` × `Thw_Tc`):

| Pb_Pc \ Thw_Tc | 1.1 (Tcw_Tc 0.95) | 1.2 (Tcw_Tc 0.90) | 1.4 (Tcw_Tc 0.80) |
|---|---|---|---|
| 1.5 | case01 | case02 | case03 |
| 2.0 | case04 | case05 | case06 |
| 5.0 | case07 | case08 | case09 |

Grid: 96×128×96 (Nx×Ny×Nz), 400 snapshots/case. `Re_b` (half-height
convention, `DELTA` = 175 µm) spans ~1450–2120 across the grid (see memory
`transcritical-series-reb-convention`); `Pr_b` ~1.4–2.7; `Ma_b` < 0.003
throughout (near-incompressible).

**Held-out OOD/validation — case10, case15** (both `Pb_Pc` = 1.5, i.e. same
pressure as case01–03, but `Thw_Tc`/`Tcw_Tc` off the training grid: case10 =
1.185/1.035, case15 = 1.132/0.982 — intermediate thermal forcing not sampled
during training).

## Splits

- **Interpolation set**: leave-one-case-out (LOCO) within case01–case09 — fit
  the representation/clustering on 8 cases, evaluate whether the held-out
  case's feature vector lands in the cluster matching its true
  `(Pb_Pc, Thw_Tc)` cell.
- **OOD / extrapolation set**: case10 and case15 — thermal-condition
  interpolation at fixed pressure (roadmap v2 §9.5), *not* pressure
  extrapolation (no case exists outside `Pb_Pc` ∈ [1.5, 5.0] yet) and *not*
  a thermodynamic-regime shift in the pseudo-boiling sense (§9.6) — that
  split isn't populated by the current database and is out of scope until
  new cases are added.
- Random split (§9.1) is not used as evidence of generalization, only as a
  sanity baseline if needed.

## Target variables

Regime discovery operates on *derived physical features*, not raw fields.
Per-case feature vector (Track A "standard physics layer" output, to be
built per the execution plan in `PROJECT_CONTEXT.md`):

- wall-normal mean profiles: `avg_u`, `avg_T`, `avg_rho` (and derived
  `avg_c_p`, `avg_mu`, `avg_kappa`) — reduced via the existing 1D-profile
  convention (x/z-averaged over interior cells).
- wall-normal RMS profiles: `rmsf_u`, `rmsf_T` (turbulence intensity).
- bulk dimensionless groups: `Re_b`, `Pr_b`, `Ec_b`, `Br_b`, `Ma_b`,
  `Re_tau_cw`, `Re_tau_hw` (already implemented, reuse rather than
  reimplement).
- POD energy spectrum (leading-mode energy fractions) as a modal-space
  feature, once the POD/SPOD module exists.

## Baseline methods

- POD/SPOD (existing group expertise, already implemented in
  `pub4_grassmann_rom/src/pod.py`) — conventional reference the learned
  representation must be compared against, not merely a competitor.
- PCA / linear projection of the feature vector, as the simplest possible
  "does linear structure already separate the regimes" baseline before any
  nonlinear ML is justified.
- k-means / agglomerative clustering on both the PCA baseline and any
  learned representation, for regime recovery.

## Compute budget

Local workstation only for Phase 0–2 (feature extraction from existing
snapshots + PCA/clustering is cheap — no training loop, no GPU). HPC/GPU is
deferred until Track A reaches Phase 4+ (advanced ML models), which is not
scheduled to start before Phase 2 produces a result.

## Evaluation metrics

**Primary — per-axis nearest-centroid diagnostics, evaluated
independently:**

- Pressure (`Pb_Pc`): LOCO nearest-centroid accuracy on the `bulk`
  dimensionless-group block alone (`Re_b`, `Pr_b`, `Ec_b`, `Br_b`, `Ma_b`,
  `Re_tau_cw`, `Re_tau_hw`) — the block `FINDINGS.md` §3 found actually
  carries the pressure signal (ARI 0.36, LOCO 0.56, best of every feature
  set tried).
- Thermal (`Thw_Tc`): LOCO nearest-centroid accuracy on the `rms_profile`
  block alone (full `rmsf_u`/`rmsf_T` wall-normal profiles, not their
  peak) — the block that carries the thermal signal (ARI 0.48, LOCO 0.78,
  best of every feature set tried).
- Each axis is tested on its own specialist block, not a shared
  representation — see "why a shared clustering isn't primary evidence"
  below for why this is deliberate, not a workaround.
- `mean_profile` (full `avg_u`/`avg_T`/`avg_rho` profiles) and `pod`
  (POD energy fractions) are excluded here: `mean_profile` was
  uninformative for either axis (ARI 0.000/0.000, `FINDINGS.md` §3) and
  `pod` measurably hurts the thermal signal when combined with
  `rms_profile` (`FINDINGS.md` §4) without adding anything `bulk` doesn't
  already do better for pressure.

**Why a single shared clustering is not used as primary evidence.**
case01–case09 is a 3×3 factorial design: `Pb_Pc` and `Thw_Tc` are two
independent 3-level partitions of the same 9 cases. A single k=3
clustering is *one* partition of those 9 points — it can align with the
`Pb_Pc` partition, the `Thw_Tc` partition, or some blend of both, but it
cannot equal two different partitions of the same set simultaneously
unless they happen to coincide, which orthogonal experimental-design axes
structurally don't (confirmed empirically in `FINDINGS.md` §4: the best
combined-space clustering found visibly blends both axes rather than
cleanly recovering either). This is a ceiling on cluster-agreement metrics
specifically, not evidence the underlying feature space lacks structure.

**Secondary — combined-space sanity check, not primary evidence for H1/H2:**

- MFA block-weighted combination of `bulk` + `rms_profile`
  (`metis.evaluation.regime.combine_blocks_mfa`, standardize each block
  then divide by its own leading singular value so raw dimension count
  stops determining influence) checked for whether the merged space
  "blends sensibly" — its k=3 clusters should track a mix of both axes
  rather than noise. Confirmed in `FINDINGS.md` §4. Neither axis's
  ARI/LOCO in this combined space beats its own specialist block, so it's
  a plausibility check, not a stronger result than the primary metrics.
- Cluster-to-known-regime agreement (adjusted Rand index between a shared
  k=3 clustering and each axis's true level) is kept only as this
  secondary combined-view diagnostic, given the ceiling above.
- Qualitative (H3): does case10/case15's position in feature space match
  the `Pb_Pc=1.5` training regime, not a distinct one? Confirmed on both
  the `bulk` specialist alone and the MFA-combined view (`FINDINGS.md`
  §1, §4); breaks under naive concatenation with `rms_profile` or the
  full rich vector (`FINDINGS.md` §2, §4) — itself informative about
  which feature choices are safe for OOD generalization, independent of
  the H1/H2 question.

Modal-subspace distance against POD subspaces (principal angles,
originally proposed here) is dropped from the evaluation plan: POD energy
fractions were tested directly as a feature block and didn't carry usable
regime signal for this task (`FINDINGS.md` §3–4), so there's no basis yet
for treating POD-subspace distance as a meaningful diagnostic here. This
is a statement about fit for *this* task, not about POD/SPOD's validity
as the group's reference modal methodology in general.

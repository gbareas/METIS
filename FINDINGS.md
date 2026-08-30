# Findings

Running log of results from the reference analyses defined in
`research_protocol.md`. Numbered, dated, most recent last.

## 1. Regime-discovery reference task, first pass (2026-08-29)

Pipeline: `scripts/run_regime_discovery.py`, results in
`results/regime_discovery.json`.

Feature vector (14-dim, `metis.features.regime`): 7 bulk dimensionless
groups (Re_b, Pr_b, Ec_b, Br_b, Ma_b, Re_tau_cw, Re_tau_hw) + 2 RMS-profile
peak scalars (rmsf_u, rmsf_T) + 5 leading POD energy fractions at the
channel-center homogeneous plane (s3_center, field u). 11 cases: case01-09
(3x3 train grid in Pb_Pc x Thw_Tc), case10/case15 (OOD, same Pb_Pc=1.5,
off-grid Thw_Tc). Standardized, PCA, k=3 k-means, evaluated against
`research_protocol.md`'s H1/H2/H3.

**H1 (pressure axis cleanly separable from bulk+POD features): NOT
confirmed as stated.** Unsupervised k=3 clustering gives ARI vs Pb_Pc =
0.226 (weak positive, far from the 1.0 "cleanly separable" claim). LOCO
nearest-centroid accuracy on the Pb_Pc axis = 4/9 (44%), barely above the
33% chance baseline for 3 classes. The k=3 clusters don't respect the
pressure grid: {case01,02,03,05,06} / {case04,08,09} / {case07} — case05
and case06 (Pb_Pc=2.0) fall in with the Pb_Pc=1.5 group, and case04
(Pb_Pc=2.0) falls in with the Pb_Pc=5.0 group.

**H2 (thermal axis harder to separate): confirmed, more strongly than
expected.** ARI vs Thw_Tc = -0.161 (worse than random labeling). LOCO
accuracy = 3/9 (33%, exactly chance). This feature set carries no
recoverable Thw_Tc signal at all, not just a weaker one.

**H3 (case10/case15 land in the Pb_Pc=1.5 regime, not a distinct one):
confirmed.** Both OOD cases' nearest Pb_Pc training centroid (in PCA
space) is 1.5, matching their true value exactly (2/2).

**Interpretation.** H1 was likely overoptimistic given how compact this
first-cut feature set is: reducing each case to 7 bulk scalars + 2 RMS
peaks + 5 POD fractions from a single slice throws away most of the
spatial structure that plausibly encodes the pressure regime (e.g. the
full wall-normal profile shape, not just its RMS peak). The bulk
dimensionless groups alone appear to not linearly separate the 3x3 grid
at n=9 — PC1 (57% of variance) doesn't cleanly track Pb_Pc.

**Next, before concluding the pressure axis is fundamentally
unrecoverable from this class of features:**
- Try the full wall-normal mean/RMS profile vectors (not just a peak
  scalar) as features — all cases share the same 96x128x96 grid, so
  profiles are directly comparable across cases without interpolation.
- Try POD energy fractions from additional slice locations (near-wall
  s5/s8, not just s3_center) and/or additional fields (T, cp not just u).
- Re-run the same H1/H2/H3 evaluation on the enriched feature set before
  revising the hypotheses themselves.

## 2. Regime-discovery reference task, enriched feature set (2026-08-29)

Pipeline unchanged; `metis.features.regime.build_feature_vector_rich`
(~677 features: full `avg_u`/`avg_T`/`avg_rho` and `rmsf_u`/`rmsf_T`
profiles, interior points, + POD energy fractions from 3 slice locations
x 2 fields) run through the exact same standardize/PCA/k-means/ARI/LOCO
pipeline as §1's compact (14-feature) set, side by side
(`results/regime_discovery.json`, both under one file now).

| metric | compact (14 feat.) | rich (677 feat.) |
|---|---|---|
| ARI vs Pb_Pc | 0.226 | **-0.118** |
| ARI vs Thw_Tc | -0.161 | **0.353** |
| LOCO Pb_Pc | 0.444 | 0.333 (chance) |
| LOCO Thw_Tc | 0.333 (chance) | 0.444 |
| case10 -> nearest Pb_Pc centroid | 1.5 (correct) | 2.0 (**wrong**, true 1.5) |
| case15 -> nearest Pb_Pc centroid | 1.5 (correct) | 1.5 (correct) |

**The enrichment did not uniformly improve regime discovery — it traded
axes.** Under the rich feature set, the pressure axis (H1) gets *worse*
than the compact set (ARI goes from weakly positive to negative; LOCO
drops to exact chance), while the thermal axis (H2) gets meaningfully
better (ARI 0.353, still far from clean but no longer negative). The
rich-feature k=3 clustering ({case01,02,04,05,07,08} / {case03,06} /
{case09}) is visibly organized around `Thw_Tc=1.4` vs. lower `Thw_Tc`, not
around `Pb_Pc`. H3 also partially breaks: case10 now lands nearest the
wrong (`Pb_Pc=2.0`) training centroid.

**Interpretation — likely a feature-counting artifact, not evidence that
pressure structure disappeared.** The rich vector is ~640 profile-point
features (dominated by `avg_T`/`rmsf_T`, which plausibly vary more by
`Thw_Tc` than by `Pb_Pc`) plus only 7 bulk scalars and 30 POD fractions.
Per-feature standardization gives every one of those ~677 dimensions
equal weight before PCA, so an axis that happens to modulate *more
individual profile points* (thermal) can dominate the leading principal
components purely by outnumbering the axis that modulates a handful of
scalars strongly (pressure) — independent of which physical effect is
actually larger. This is a standard pitfall of concatenate-then-PCA
feature engineering, not a physical conclusion.

**Next**: don't add more raw features — instead run a per-block ablation
(bulk-only vs. mean-profile-only vs. RMS-profile-only vs. POD-only
through the same pipeline) to identify which block actually carries the
pressure signal vs. the thermal signal, before trying any block-weighted
or block-wise-PCA combination. That's a cleaner diagnostic than guessing
at reweighting schemes.

## 3. Regime-discovery reference task, per-block ablation (2026-08-29)

Pipeline: `scripts/run_regime_ablation.py`, results in
`results/regime_discovery_ablation.json`. Each of the four blocks behind
the rich feature set — `bulk` (7 feat.), `mean_profile` (384 feat.:
`avg_u`/`avg_T`/`avg_rho`), `rms_profile` (256 feat.: `rmsf_u`/`rmsf_T`),
`pod` (30 feat.: 3 slices x 2 fields x 5 modes) — run through
`metis.evaluation.regime.evaluate_regime_discovery` **in isolation**.

| block | n feat | ARI Pb_Pc | ARI Thw_Tc | LOCO Pb_Pc | LOCO Thw_Tc | OOD nearest Pb_Pc |
|---|---|---|---|---|---|---|
| bulk | 7 | **0.357** | 0.071 | **0.556** | 0.111 | 1.5, 1.5 (both correct) |
| mean_profile | 384 | 0.000 | 0.000 | 0.444 | 0.333 | 2.0, 2.0 (both wrong) |
| rms_profile | 256 | -0.161 | **0.484** | 0.222 | **0.778** | 5.0, 5.0 (both wrong) |
| pod | 30 | 0.353 | -0.118 | 0.222 | 0.111 | 5.0 (wrong), 1.5 (correct) |

**This resolves §2's open question cleanly: the two signals live in
different, specific blocks, not spread evenly.**

- **The pressure signal lives in `bulk`.** ARI 0.357 and LOCO 0.556 are
  both the best pressure results across every feature set tried so far
  (better than compact's own 0.226/0.444, since compact also carries 2
  RMS-peak scalars and 5 POD fractions that evidently pull the pressure
  signal in the wrong direction when mixed in). Both OOD cases land on
  the correct `Pb_Pc=1.5` centroid — H3 is best supported using `bulk`
  alone. Physically sensible: `Re_b`, `Ma_b`, etc. are dominated by
  pressure-driven thermophysical-property changes.
- **The thermal signal lives in `rms_profile`.** ARI 0.484 and LOCO 0.778
  are both the best thermal results by a wide margin — confirms the §2
  suspicion that `rmsf_T`'s full profile shape (not `mean_profile`, which
  is essentially uninformative for either axis: ARI exactly 0.000 on
  both) is what actually encodes `Thw_Tc`.
- **`mean_profile` is close to uninformative for this grid** (ARI 0.000
  vs. both axes) despite being the single largest block (384 of 677
  features in the rich set) — it was diluting §2's combined result
  without contributing signal for either hypothesis.
- **`pod` (at these 3 slice locations, fields u/T) partially tracks
  pressure** (ARI 0.353, close to `bulk`'s) but its LOCO/OOD numbers are
  weak and inconsistent with its own ARI, suggesting the unsupervised
  k=3 partition happens to align with `Pb_Pc` more than a per-axis
  nearest-centroid rule would justify — treat this block's contribution
  as unreliable until re-tested with more slices/fields or a larger `k`.

**This also explains §2's "trade" directly**: the rich set's 677
dimensions are ~57% `mean_profile` (uninformative dead weight), ~38%
`rms_profile` (strong thermal signal), and only ~5.5% `bulk`+`pod`
(the two blocks that actually carry pressure) — so naive concatenation
with per-feature standardization was always going to make the rich set
look thermal-dominated, independent of the physics.

**Next**: this is now well-understood enough to combine deliberately
rather than guess. The natural next step is a block-wise PCA (reduce
each block to a small number of components *before* concatenating,
e.g. via Multiple Factor Analysis / MFA-style block normalization, so
`bulk`'s 7 features and `mean_profile`'s 384 get comparable influence
regardless of raw dimension count) using `bulk` + `rms_profile` only
(drop `mean_profile` as uninformative, keep `pod` provisional pending
further testing) — expect this to recover both H1 and H2 simultaneously
where no single block or naive concatenation could. Not started.

## 4. Regime-discovery reference task, block-wise MFA combination (2026-08-29)

Pipeline: `scripts/run_regime_blockwise.py`, results in
`results/regime_discovery_blockwise.json`. Added
`metis.evaluation.regime.mfa_normalize_block`/`combine_blocks_mfa`
(standardize each block, then divide by its own leading singular value,
so every block's leading axis has equal inertia before concatenating —
classic MFA block weighting) plus a `standardize_input=False` escape
hatch on `evaluate_regime_discovery` so its default per-feature
standardization doesn't undo that weighting. Validated in isolation first
(`tests/unit/test_evaluation_regime.py`): on a synthetic 2-feature signal
next to 40 unstructured noise features, naive concatenation gives ARI
-0.07 (destroyed), MFA weighting recovers 0.35 (signal-only, no noise,
gets 1.0 — MFA narrows the gap, doesn't erase it). That result already
told us not to expect a perfect fix on the real blocks either.

| combination | method | ARI Pb_Pc | ARI Thw_Tc | LOCO Pb_Pc | LOCO Thw_Tc | OOD nearest Pb_Pc |
|---|---|---|---|---|---|---|
| bulk + rms_profile | naive | -0.161 | 0.484 | 0.222 | 0.778 | 5.0, 5.0 (both wrong) |
| bulk + rms_profile | **mfa** | 0.071 | 0.357 | 0.444 | 0.444 | **1.5, 1.5 (both correct)** |
| bulk + rms_profile + pod | naive | -0.161 | 0.484 | 0.222 | 0.667 | 5.0, 5.0 (both wrong) |
| bulk + rms_profile + pod | mfa | 0.226 | -0.161 | 0.333 | 0.222 | 1.5, 1.5 (both correct) |

*(for reference, §3: `bulk` alone = ARI 0.357/LOCO 0.556 on Pb_Pc;
`rms_profile` alone = ARI 0.484/LOCO 0.778 on Thw_Tc.)*

**MFA weighting is a real, measurable fix for the dilution problem — but
it does not deliver "both hypotheses confirmed by one clustering."**
Concretely: naive `bulk+rms_profile` is just `rms_profile` again (its 256
features fully dominate `bulk`'s 7, identical ARI/LOCO on Thw_Tc to the
bare `rms_profile` block in §3) and gets H3 wrong for both OOD cases.
MFA weighting fixes H3 outright (both OOD cases correctly nearest
`Pb_Pc=1.5` again) and pulls the clustering into a real mix of both axes
— inspecting the k=3 assignment
(`cluster0={case03,06,09}` = exactly the three `Thw_Tc=1.4` cases;
`cluster2={case07,08}` = 2 of the 3 `Pb_Pc=5.0` cases) shows it is
genuinely blending both signals, not just picking one. But neither ARI
nor LOCO on either axis reaches what that axis's *specialist* block
achieved alone (bulk: ARI 0.357/LOCO 0.556 on Pb_Pc; rms_profile: ARI
0.484/LOCO 0.778 on Thw_Tc) — combining cost each axis some of its own
best performance rather than adding up.

**Why, structurally, "one clustering confirms both H1 and H2" was
probably never achievable.** case01-09 is a 3x3 factorial design:
`Pb_Pc` and `Thw_Tc` are two *independent* 3-level partitions of the
same 9 points. A single k=3 clustering is one partition of those 9
points — it can align closely with the `Pb_Pc` partition, or the
`Thw_Tc` partition, or some blend of both (as the mixed cluster above
shows), but it cannot equal two different partitions of the same set
simultaneously unless they happen to coincide, which orthogonal
experimental-design axes structurally don't. This is a ceiling on the
**ARI-vs-single-k=3-clustering** metric specifically, not necessarily on
the feature space itself. The **LOCO nearest-centroid** metric doesn't
have this structural ceiling (it tests each axis independently via its
own per-axis centroids, not a shared clustering) — but empirically it
still landed below the specialist blocks here (0.444/0.444 vs. the
specialists' 0.556/0.778), so the shortfall isn't purely a metric
artifact; there's likely real geometric interference between the two
blocks in the shared PCA space too.

**`pod`'s provisional status is resolved: drop it.** Adding it to the
MFA combination pushes ARI further toward pressure (0.226, closer to
`bulk`'s own 0.357) at the cost of collapsing the thermal signal
entirely (ARI Thw_Tc -0.161, LOCO Thw_Tc 0.222) — consistent with `pod`
itself leaning pressure in §3 (ARI 0.353) with an unreliable LOCO. It
adds no new information `bulk` doesn't already provide better, and
actively hurts `rms_profile`'s contribution when combined.

**Revised framing for `research_protocol.md`'s primary task**: a single
unified unsupervised clustering over one combined feature space is
probably the wrong instrument for recovering *both* the pressure and
thermal regime axes at once, even with the dilution problem fixed — the
two axes are better recovered by two axis-specific diagnostics (`bulk`
dimensionless groups for pressure, `rms_profile` turbulence-intensity
shape for temperature) than by one global method. This refines rather
than fails H1/H2: both are now positively supported *individually*, by
their respective specialist block, with a mature, understood reason why
forcing them into one clustering doesn't compound the wins. Recommend
updating `research_protocol.md`'s evaluation section to reflect
axis-specific diagnostics as the primary method, with the combined/MFA
view kept only as a secondary "does the space at least blend sensibly"
check (which §4 confirms it does, per the mixed-cluster inspection
above).

## 5. Regime discovery frozen as the `regime-v1` benchmark (2026-08-30)

§1-4 are done. The conclusion is now a platform acceptance test, not an
open line of tuning:

- Spec: `configs/benchmarks/regime_discovery_v1.yaml` (train case01-09,
  OOD case10/case15; `bulk` = pressure diagnostic, `rms_profile` =
  thermal diagnostic; MFA `bulk`+`rms_profile` as the combined sanity
  check; every §3/§4 metric pinned with a 1e-6 tolerance).
- Runner: `metis.evaluation.benchmark.run_regime_v1` +
  `scripts/benchmark_regime_v1.py` (writes `results/regime_v1.json`,
  exits non-zero on any regression). Reproduces §3/§4 exactly against
  the real data — 7/7 checks pass.
- Regression test: `tests/regression/test_regime_v1_benchmark.py`, run
  on a 60 KB committed feature artifact
  (`tests/data/regime_v1/block_features.npz`) so CI needs no DNS. It
  asserts `bulk` stays the best block on both pressure metrics,
  `rms_profile` on both thermal metrics, MFA restores the OOD
  `Pb_Pc=1.5` assignment where naive concatenation gets it wrong, and no
  metric drifts.

Changing any expected value in the config now means the science changed
and needs its own FINDINGS entry.

## 6. Autoencoder representation learning on case-level vectors — stop (2026-08-30, corrected 2026-08-31)

**I2-A** (`scripts/representation_study.py`, MLflow experiment
`representation-v1`, results in `results/representation_study.json`).

Question: does a nonlinear autoencoder latent organise the 9 training
cases (and place the 2 OOD cases) by pressure/thermal any better than
2-component PCA of the same features? Blocks `bulk` (7 feat) and
`rms_profile` (256 feat), train-only standardisation, latent dim 2,
Autoencoder(hidden=32, tanh) over 5 genuinely independent seeds.

> **Correction (2026-08-31).** The first pass reported `± 0.000` seed
> variance and `latent_stability ~ 1e-16`, and concluded "the AE just
> finds the linear subspace". That was an artefact: `_make_net` reset
> torch to seed 0 *inside* network construction, so every nominal seed
> started from identical weights. Fixed (the experiment seed now drives
> weight init) and rerun. The **headline conclusion survives** — the AE
> does not beat PCA on any block — but the reasoning changes.

**Result: no — across independent initialisations the AE does not
robustly improve on PCA, and is markedly more initialisation-sensitive.**

| block | metric | PCA(2) | Autoencoder(2), mean ± std over 5 seeds |
|---|---|---|---|
| bulk | ARI vs Pb_Pc | 0.357 | 0.353 ± 0.000 |
| bulk | ARI vs Thw_Tc | 0.071 | −0.118 ± 0.000 |
| rms_profile | ARI vs Pb_Pc | −0.161 | −0.034 ± 0.108 |
| rms_profile | ARI vs Thw_Tc | **0.484** | 0.237 ± 0.228 |

- **`rms_profile`**: the AE's thermal-axis ARI is both **worse on
  average** (0.237 vs PCA's 0.484) and **highly seed-dependent**
  (std 0.228). `latent_stability` (Procrustes-aligned relative L2 across
  seeds) is 0.12 — the embedding genuinely varies from run to run.
- **`bulk`**: only 7 features into a 2-D latent; the k=3 cluster
  assignment lands on the same ARI every seed (std 0), but the latent
  itself still varies (`latent_stability` 0.23), and the ARIs sit below
  PCA's.
- `beats_baseline` is **False on every block** (`assess_model`): where an
  ML metric improves, a physical/organisation metric regresses.

**Per the I2 stop criterion (§14): recorded and parked.** Case-level
representation learning is not pursued further — with only 9 training
cases the nonlinear model has no robust advantage and adds
initialisation risk. Escalating to **I2-B** (2D slice fields, N ~
thousands of snapshots per case rather than one vector) is the
recommended next scientific step (updated plan §24–29), but it is a
different task and a deliberate decision. The `PCARepresentation` /
`Autoencoder` / `Trainer` / `evaluate_representation` machinery is
reusable there.

## 7. I2-B slice-level representation learning — negative result, stopped (2026-08-30)

**Verdict (B7): H-I2B falsified.** A fixed conv-autoencoder on centre-plane
`u'` snapshots does **not**, at any tested latent dim, beat a linear
PCA/POD basis on reconstruction, OOD robustness, physical fidelity, or
latent interpretability — and its latent is seed-unstable while PCA's is
exact. `assess_model` returns `is_better=False` for **all 5 seeds at
every k**. Linear representations are sufficient for this flow structure
under these thermodynamic conditions. **Stopped** — no VAE / U-Net / FNO
/ operator escalation, no model registered (`results/i2b_decision.json`).

Protocol: `docs/i2b_representation_protocol.md` (frozen 2026-08-30, B1).
Dataset: `metis dataset build-slices` primary split — centre-plane (`s3_center`)
streamwise-velocity fluctuation snapshots, 96×96, train 3600 (case01-09
snaps 0-399) / val 900 (same cases 400-499, chronological) / OOD 1000
(all of case10, case15). Standardised by one global mean/std fit on train
(mean 0.015, std 0.073).

### B3 — PCA / snapshot-POD baseline (2026-08-30)

`scripts/i2b_baseline.py`, results in `results/i2b_baseline.json`, MLflow
`i2b-representation/baseline-pca`.

- **PCA ≡ method-of-snapshots POD**: on the same centred training matrix,
  max singular-value relative difference **2e-15** — the two are the same
  computation to machine precision, as expected.
- **Linear reconstruction is high-rank.** Cumulative reconstructed
  variance on the training split:

  | k | var | val relL2 | OOD relL2 |
  |---|---|---|---|
  | 2 | 0.233 | 0.938 | 0.525 |
  | 4 | 0.350 | 0.875 | 0.453 |
  | 8 | 0.473 | 0.824 | 0.435 |
  | 16 | 0.630 | 0.729 | 0.393 |
  | 32 | 0.776 | 0.647 | 0.349 |

  The protocol's "smallest k with ≥90% reconstructed variance" rule has
  **no answer in the grid** — 32 modes capture only 77.6%. Frozen
  `headline_latent_dim = 32` (largest grid dim); the AE-vs-PCA comparison
  runs across the whole grid regardless. That a 96×96 turbulent
  fluctuation field needs > 32 linear modes for 90% is the expected
  behaviour of broadband turbulence, and is exactly the regime where a
  nonlinear model *could* help (H-I2B) — or where the field is simply
  irreducibly high-rank. B5-B7 decides.
- **OOD relL2 < val relL2 at every k.** case10/case15's `u'` fields are
  reconstructed *better* by the train PCA basis than the training cases'
  own held-out snapshots. Likely because both OOD cases sit at Pb_Pc=1.5
  (matching case01-03) and their fluctuation energy, once standardised by
  the train-wide scale, is lower and more concentrated in the leading
  modes. Not alarming; noted for interpretation when the AE OOD numbers
  come in.

### B5 — conv-autoencoder training, seed × latent-dim grid (2026-08-30)

`scripts/i2b_train.py`, manifest in `results/i2b_training.json`, 25 MLflow
runs `i2b-representation/convae-k*_seed*` (tag `phase=B5`), best-state
checkpoints under
`artifacts/models/i2b_representation_v1_primary_u_s3_center/`. Fixed
architecture per protocol §4 (Conv 1→16→32→64, stride 2, GELU; linear
bottleneck; mirrored ConvTranspose), `Trainer` defaults untouched (Adam
lr 1e-3, batch 64, early stop on val MSE, patience 150), CUDA, ~57 s/run.
**Training only — the reconstruction / physical / latent evaluation and
the `assess_model` gate are B6-B7.**

- **Best val MSE (standardised field), mean ± std over 5 seeds:**

  | k | val MSE | train MSE | best epoch | √(val MSE) ≈ relL2 | PCA val relL2 |
  |---|---|---|---|---|---|
  | 2 | 0.866 ± 0.023 | 0.775 | 1 | 0.93 | 0.938 |
  | 4 | 0.760 ± 0.012 | 0.602 | 1 | 0.87 | 0.875 |
  | 8 | 0.674 ± 0.011 | 0.544 | 1 | 0.82 | 0.824 |
  | 16 | 0.562 ± 0.008 | 0.340 | 1–2 | 0.75 | 0.729 |
  | 32 | 0.449 ± 0.011 | 0.231 | 2 | 0.67 | 0.647 |

  (√(val MSE) is a rough proxy for relL2 since the standardised field has
  ≈ unit variance; B6 computes the real `metis.evaluation.metrics`
  numbers on both val and OOD.)

- **Seed-stable.** Val-MSE seed std is 1–3 % of the mean at every k — the
  §18 seed fix holds up and the conv-AE optimisation is reproducible.

- **The conv-AE reaches its validation optimum in epoch 1–3, then only
  train MSE keeps falling** (k=32: train 0.23 vs val 0.45). With protocol
  settings (no tuning) the model captures the generalisable structure on
  its first pass and overfits pixel detail thereafter — best-state
  restore + early stopping is doing real work here.

- **Preliminary read (not the verdict):** on validation the conv-AE
  *matches* PCA at k ≤ 8 and is *slightly worse* at k = 16, 32 — no
  compressive advantage visible so far. Whether it wins on OOD
  robustness, physical fidelity, or latent interpretability is what B6-B7
  tests before the decision gate.

### B6 — §5.1–5.5 evaluation vs the PCA baseline (2026-08-30)

`scripts/i2b_evaluate.py` reloads all 25 checkpoints and scores them
against PCA (refit once at k=32, truncated per k) on the val and OOD
splits, standardised space. Full numbers in `results/i2b_evaluation.json`.
The `assess_model` accept/stop gate is B7; this entry is the metric
tables.

- **5.1 reconstruction — conv-AE loses to PCA at every k, on both
  splits.** relative-L2:

  | k | val PCA | val AE (μ±σ/5) | OOD PCA | OOD AE |
  |---|---|---|---|---|
  | 2 | 0.938 | 0.955 ± 0.013 | 0.525 | 0.770 ± 0.136 |
  | 4 | 0.875 | 0.895 ± 0.007 | 0.453 | 0.630 ± 0.044 |
  | 8 | 0.824 | 0.843 ± 0.007 | 0.435 | 0.548 ± 0.018 |
  | 16 | 0.729 | 0.769 ± 0.005 | 0.393 | 0.519 ± 0.040 |
  | 32 | 0.647 | 0.688 ± 0.008 | 0.349 | 0.455 ± 0.013 |

  PCA is the optimal linear subspace in closed form; the fixed, untuned
  conv-AE (early-stopping at epoch 1–2, §B5) never exploits nonlinearity
  enough to catch it.

- **5.2 robustness — worse on both counts.** OOD-degradation ratio
  (OOD relL2 / val relL2, lower better) at k=32: **PCA 0.54 vs AE
  0.66 ± 0.02**. `latent_stability` (Procrustes relL2 of the latent
  across the 5 seeds, 0 = identical): **AE max 0.38 at k=32, 0.65–2.2 at
  smaller k; PCA is exactly 0.** The AE latent is not reproducible across
  seeds — this alone matches a protocol §6 stop criterion.

- **5.3 physical fidelity — tie on val, AE worse on OOD.** k=32 val:
  RMS-profile relL2 0.243 (AE) vs 0.244 (PCA), x-spectrum relL2 0.347 vs
  0.354 — a wash, except the AE's **log-spectrum correlation is 0.80 vs
  PCA's 1.00** (it distorts spectral shape). k=32 OOD: RMS-profile relL2
  **0.257 (AE) vs 0.129 (PCA)**, x-spectrum relL2 **0.356 vs 0.070** —
  PCA reconstructs unseen-condition physics far better. POD
  energy-fraction L1 is comparable throughout.

- **5.4 latent interpretation — comparable, no AE edge.** Max |corr| of
  any latent axis vs (Pb_Pc, Thw_Tc, per-snapshot RMS, spectral-peak k)
  at k=32 val: AE ≈ (0.66, 0.66, 0.69, 0.63) vs PCA ≈ (0.61, 0.61, 0.77,
  0.48). The AE is marginally higher on the operating-point variables and
  peak-k, lower on snapshot RMS — within the noise of an unstable latent.

- **Direction: negative.** The conv-AE is not more compressive, not more
  OOD-robust, not more physically faithful, and not more interpretable
  than PCA/POD at matched latent dim; its latent is seed-unstable while
  PCA's is exact. B7 runs `assess_model` per seed to formalise the
  accept/stop decision.

### B7 — accept/stop gate: STOP (2026-08-30)

`scripts/i2b_decision.py`, `results/i2b_decision.json`, MLflow
`i2b-representation/decision` (tag `phase=B7`). `assess_model` per seed
(protocol §17 rule: an ML metric must improve **and** no physical
diagnostic may regress), candidate = conv-AE seed, baseline = PCA at the
same k.

- `ml_metrics` = `val_relative_l2`, `ood_relative_l2`,
  `ood_degradation_ratio` (lower-is-better for all three).
- `physical_metrics` = mean/RMS-profile relL2, x/z spectrum relL2, x/z
  log-spectrum correlation, POD energy-fraction L1 — on both val and OOD.

**Result — `is_better = False` for all 5 seeds at every k (2/4/8/16/32).**
At the headline k=32 the per-seed ML deltas (signed, + = better) are
about `val_relative_l2 −0.04`, `ood_relative_l2 −0.09`,
`ood_degradation_ratio −0.10` — no ML metric improves, so the gate never
even reaches the physical check. For the record it would fail there too:
the AE regresses ~16 of the 18 physical diagnostics per seed
(log-spectrum correlation −0.2 to −0.3, OOD spectrum relL2 −0.22),
improving only POD energy-fraction L1.

Two protocol §6 negative criteria are met: (1) no seed reaches
`is_better=True`; (2) `latent_stability` is poor — AE Procrustes relL2
across seeds 0.38 (k=32) to 2.16 (k=4) while PCA's latent is exact.

**Decision: NEGATIVE — linear PCA/POD is sufficient for centre-plane
`u'` structure across the tested thermodynamic conditions. Stop.** No
escalation to VAE / U-Net / FNO / DeepONet; no model registered. I2-B-2
(`T` field) and near-wall slices are *not* triggered — this is a clean
negative, not an inconclusive one. B8 assembles the modal/physical
comparison from `results/i2b_evaluation.json`; B9 is the final writeup +
a `metis report` for the study.

### B8 — modal & physical-statistic comparison (2026-08-30)

`scripts/i2b_modal_compare.py`, `results/i2b_modal_compare.json`, figures
under `reports/i2b-representation/figures/` (regenerated, not committed).
PCA at k=32 vs the best conv-AE seed (seed 3) at k=32.

- **The field is irreducibly high-rank.** The training POD spectrum
  needs **67 modes for 90 %** and **181 for 99 %** of the variance; k=32
  captures 77.6 %. There is no compact linear code — and, per B6/B7, no
  compact nonlinear one either at k ≤ 32.

- **The conv-AE spans essentially the same leading subspace as POD, just
  less accurately.** Leading POD energy fractions of the *reconstructed*
  val ensembles are nearly identical for the two models — mode 1: true
  0.163, PCA 0.261, AE 0.258; top-5 cumulative: true 0.49, PCA 0.70, AE
  0.68. Both reconstructions over-concentrate energy into the leading
  modes (a low-rank projection always does — it drops the broadband
  tail), and the AE's modal signature tracks PCA's to within ~0.01. The
  nonlinearity is not buying a different representation.

- **Where the AE actually loses is fidelity, especially OOD.** Ensemble
  wavenumber-spectrum relative-L2 (dx = index units): val x-spectrum
  PCA 0.354 vs AE 0.384; **OOD x-spectrum PCA 0.070 vs AE 0.224**. RMS
  profile along x: val PCA 0.244 vs AE 0.267; **OOD PCA 0.129 vs AE
  0.222**. On unseen operating conditions the linear basis reconstructs
  the second-order statistics ~3× better than the conv-AE.

- **Reading.** POD gives the optimal rank-k subspace in closed form; the
  fixed, untuned conv-AE (which early-stops at epoch 1–2, §B5)
  approximates that same subspace and adds reconstruction error and
  spectral distortion on top. That is the whole mechanism behind the
  negative result — there is no missing nonlinear structure for a bigger
  model to chase, so the escalation is correctly declined.

### Artifacts & reproduction (B9 — Phase B closed 2026-08-30)

Protocol `docs/i2b_representation_protocol.md` + config
`configs/experiments/i2b_representation_v1.yaml` (both frozen B1). Run
order:

| step | script | output |
|---|---|---|
| B3 | `scripts/i2b_baseline.py` | `results/i2b_baseline.json` |
| B5 | `scripts/i2b_train.py` | `results/i2b_training.json` + 25 checkpoints + MLflow `phase=B5` |
| B6 | `scripts/i2b_evaluate.py --data-root <dns>` | `results/i2b_evaluation.json` |
| B7 | `scripts/i2b_decision.py` | `results/i2b_decision.json` + MLflow `phase=B7` |
| B8 | `scripts/i2b_modal_compare.py` | `results/i2b_modal_compare.json` + `reports/i2b-representation/figures/` |
| B9 | `metis report i2b-representation --from results/` | `reports/i2b-representation/{summary.md,metrics.json,figures/}` |

The `metis report i2b-representation` generator
(`metis.reporting.generators.i2b_representation_report`) bundles the five
`results/i2b_*.json` into one standing report; `evaluation` + `decision`
are required, the rest enrich it. **No model registered — negative
result.** Phase B is closed; next is Phase C (virtual-sensor time-series
ML).

---

## 8. Virtual-sensor forecasting (Phase C, running log)

Protocol: `docs/virtual_sensor_protocol.md` + config
`configs/experiments/virtual_sensors_v1.yaml`.

### C1 — sensor / variable selection: FROZEN (2026-08-30)

Turns the `processed_slices` product into a multivariate sample-step
time series. **No physical Δt** for this product (only solver iteration
count, uniform at 2500 iters/step) → sample-step forecasting only, no
Hz/frequency claims (§35).

- **5 virtual probes**, each an existing XZ slice, chosen for transcritical
  physics: `cw_buffer` (`s5_y_plus_10_cw`), `centre` (`s3_center`),
  `hw_buffer` (`s8_y_plus_10_hw`), `pseudoboiling` (`s1_max_cp_f`, the
  Widom-line plane), `max_u` (`s2_max_u`). All 5 exist for all 11 cases.
- **Channels**: per probe × field (`u`,`T`,`cp`), two plane aggregates —
  `_pmf` (plane-mean fluctuation ⟨f′⟩) and `_rms` (plane RMS √⟨f′²⟩) →
  **30 sensor channels**, 500 steps/case. Static per-case context:
  `Pb_Pc`, `Thw_Tc`, `Tcw_Tc`, per-probe `y_loc`.
- **Task**: multivariate forecast `X_{t-31:t} → X_{t+1:t+H}`,
  `H ∈ {1,4,8,16}` steps, windows stride-1 within a case.
- **Splits** (chronological, like `regime-v1`): train case01–09 steps
  0–399; val case01–09 steps 400–499; OOD case10+case15 all steps;
  secondary pressure-holdout train 01–06 / OOD 07–09.
- Realizability checked: all 5×3×2 channels build cleanly for train and
  OOD cases; `T_rms` already resolves the cw/hw thermal asymmetry
  (case01 2.83 vs 3.77) and shifts under OOD forcing — there is signal.

### C2 — temporal dataset builder (2026-08-30)

`metis.data.datasets.build_virtual_sensor_dataset` + `metis dataset
build-sensors --experiment-config configs/experiments/virtual_sensors_v1.yaml`.
`compute_sensor_channels` reduces each probe plane to `_pmf` / `_rms`
per field; the builder concatenates the 5 probes, applies the
chronological splits, fits a per-channel train-only scaler, and emits
`{train,val,ood}.npz` (raw channels + `case_ids` + `step_idx` +
`window_anchors`) + `metadata.json` (channels, static context per case,
scaler, splits, source-manifest fingerprint) under
`artifacts/datasets/virtual_sensors_v1_primary/`.

- **primary split, real data**: 30 channels; steps train/val/ood =
  **3600 / 900 / 1000**; windows (L=32, H_max=16, stride 1, per case) =
  **3177 / 477 / 906** — exactly `n_cases·(steps − 47)`.
- **Leakage-safe by construction**: each split is built from its own
  step slice, anchors are placed per case, so no window's context or
  targets cross a case boundary or the train/val step boundary (checked
  in `tests/unit/test_virtual_sensor_dataset.py`, 7 tests; C4 pins this).
- `.assemble()` materialises `(Xc (W,32,30), Y (W,4,30), anchor_case,
  anchor_step)`; standardised train channels have mean ≈ 0, std ≈ 1.
- Fingerprint (probes + channel/task config + per-slice size/mtime
  manifest + code hash) → cache hit in 8 ms; a touched slice file busts
  it.

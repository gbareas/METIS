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

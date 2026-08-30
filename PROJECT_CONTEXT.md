# PROJECT_CONTEXT.md

## What this is

`metis` — reusable ML analysis & physical-discovery framework sitting
downstream of the group's DNS solver for high-pressure transcritical flow.
Two parallel tracks share this repo:

- **Group infrastructure (Track A / B)** — ingestion, standard physics
  layer (stats / spectra / POD / SPOD), regime discovery. Shared group
  asset; candidate for a supervised BSc/MSc thesis extension. Open-ended
  timeline.
- **Personal portfolio module (Track C / D)** — prediction task with
  OOD-gated routing between a neural-operator surrogate and the full
  solver. Wraps `pub5_neural_operators`'s trained checkpoint and OOD/LOO
  methodology directly — **decoupled from Track A/B**, does not wait on
  metis's ingestion/physics layer maturing. Built solo, for job-search
  purposes. Target: ~8-10 weeks, independent of Track A/B's timeline. See
  `docs/agent_implementation_plan.md`.

Full background: `docs/roadmap_v1_surrogate_solver.md`,
`docs/roadmap_v2_analysis_framework.md`, `docs/agent_implementation_plan.md`.

## Current status

Repo scaffolded. Ingestion layer (`metis.data.ingestion.hdf5_reader`)
wired to the real RHEA snapshot layout (flat HDF5, ghost cells,
companion `data/processed/case{NN}/metadata.json`) and verified against
real DNS output (`data/raw/case01/...h5`), not just the synthetic mock
(`metis.testing.mock_dns`). `research_protocol.md` Phase 0 is filled in:
case01-09 train / case10,15 OOD, reusing the Pub 4/5 series convention.

Standard physics layer, bulk quantities: `metis.features.physics`
(`wall_normal_profiles`, `bulk_quantities`, `friction_reynolds`,
`case_physics_summary`) ports the closed-form definitions from
`compute_case_setup_table.py` onto `DNSCase` — cross-checked bit-for-bit
against the published Pub 4 `case_setup_table.json` for case01 (Re_b,
Pr_b, Ma_b, Br_b, T_cw/hw, Re_tau_cw/hw all match exactly).

Standard physics layer, spectra: new ingestion path
`metis.data.ingestion.slice_reader.SliceReader` reads the XZ
homogeneous-plane slice product (`data/processed_slices/case{NN}/
{slice_id}/`, already fetched locally for all 11 cases as part of the
pub5_neural_operators work — a shared data product, not pub5-specific
code) into a `SliceCase`. `metis.features.spectra` computes 1D wavenumber
spectra (`wavenumber_spectrum`, `premultiplied_spectrum`) along x/z from
it — cross-checked via Parseval's identity against real case01/s3_center
data (rel. error ~1e-14 for u, T, cp on both axes). Only spatial spectra;
temporal/frequency spectra are out of scope (no physical timestep stored
alongside the slice snapshots, only iteration numbers).

Standard physics layer, POD: `metis.features.pod` (`pod`, `compute_pod`,
`PODResult.energy_fractions`/`mode_field`) ports `pod_slice.py`'s
method-of-snapshots algorithm and 0.99 energy-threshold convention onto
`SliceCase` — cross-checked against Pub 4's cached
`results_slices/case01/s2_max_u/u/info.json`: `r=29` and
`energy_captured=0.9900003143` match to 10 significant figures, singular
values match to float32 precision (Pub 4 stores them downcast). SPOD
(frequency-resolved) stays out of scope — no physical timestep stored
alongside the slice snapshots, only solver iteration numbers.

Standard physics layer (stats/spectra/POD) is feature-complete for the
case01-15 training/OOD set. Wall-normal (XY/ZY) slices remain explicitly
out of scope for all three modules unless a wall-normal-plane analysis is
later justified.

Regime-discovery reference task: feature sets, evaluation, per-block
ablation, and a block-wise MFA combination all run end-to-end on all 11
real cases through one shared pipeline (`metis.features.regime` +
`metis.evaluation.regime.evaluate_regime_discovery`/`combine_blocks_mfa`,
PCA/k-means/ARI/LOCO/MFA all numpy+scipy only, no new deps). **See
`FINDINGS.md` §1-4** for the full trail; the short version:

- compact (14 feat.) and rich (~677 feat.) traded axes rather than one
  beating the other (§1-2) — traced to naive per-feature standardization
  handing `mean_profile`'s 384 features equal weight to `bulk`'s 7.
- The §3 ablation found the actual signal locations: **`bulk` carries
  pressure** (ARI 0.36, LOCO 0.56, best of anything tried), **`rms_profile`
  carries thermal** (ARI 0.48, LOCO 0.78, best of anything tried),
  `mean_profile` is uninformative for either axis (dead weight), `pod`
  partially tracks pressure but inconsistently.
- §4 combined `bulk`+`rms_profile` via MFA block-weighting (fixes the
  dilution: validated on synthetic data, and on real data restores H3
  — both OOD cases correctly nearest `Pb_Pc=1.5`, which naive
  concatenation got wrong). **But it doesn't deliver "both axes confirmed
  by one clustering"** — each axis's specialist block still outperforms
  any combination on its own axis. Root cause is structural, not just a
  weighting bug: `Pb_Pc` and `Thw_Tc` are two independent 3-level
  partitions of the same 9 cases, and a single k=3 clustering can't equal
  two different partitions of the same set at once. `pod` is confirmed
  unhelpful (dropped: hurts `rms_profile`'s thermal signal when combined,
  adds nothing `bulk` doesn't already do better for pressure).

**Revised framing (§4)**: recover the two regime axes via two
axis-specific diagnostics (`bulk` for pressure, `rms_profile` for
thermal), not one unified clustering — this positively confirms H1 and H2
individually rather than "failing" the reference task. `research_protocol.md`'s
evaluation section has been rewritten to match.

**Track C/D, M0-M5 (2026-08-29) — complete.** `src/metis/router/` is a working
end-to-end router agent. `confidence.py` — rule-based regime-envelope/
`Tcw_Tc` diagnostic, reusing the *validated* Pub 5 finding that blind OOD
error tracks `Tcw_Tc` crossing the pseudo-critical boundary, not distance
in the surrogate's own `(Pb_Pc, Thw_Tc)` conditioning (a different
latent-geometry diagnostic was tried in Pub 5 and explicitly failed, and
is deliberately not reused here). `surrogate.py` — wraps the frozen
`unet_raw_ood` checkpoint, reproduces its recorded blind-OOD errors for
case10/case15 to ~3e-4 relative. `solver.py` — precomputed fallback,
matches `(Pb_Pc, Thw_Tc, Tcw_Tc)` against `case_descriptors.json`
(near-exact match, never a nearest-neighbor guess — a genuinely novel
point raises `KeyError`) and returns the actual converged DNS RMS fields
for one of the 11 simulated cases (case01-09, case10, case15 — not 15
cases, case11-14 were never simulated). `core.py` ties them into
`route()`. `agent.py` — wraps `route()` as a **single** Claude tool
(`route_case`, model `claude-opus-5`, SDK tool runner) rather than
exposing `surrogate_infer`/`confidence_score`/`solver_lookup` separately;
letting the LLM sequence those three itself would mean the LLM makes the
routing decision, which is exactly the "new unvalidated policy" the
project's own design principle rules out. **Not verified live** — no
`ANTHROPIC_API_KEY`/`ant auth login` credentials configured in this
environment, so `agent.py`'s actual LLM round trip is untested; only
`route_case`'s underlying logic is (against real cases). Needs the
`router` extra (`pip install -e ".[router]"` then
`pip install -e ../pub5_neural_operators` — see pyproject.toml for why
the second step is required; re-run it after any `[router]` reinstall,
it silently reverts). Real design deviation from the original plan doc:
`route_case` takes `{"Pb_Pc", "Thw_Tc", "Tcw_Tc"}` directly, not the
plan's original `inlet_pressure`/`inlet_temperature`/`mass_flow_rate`
schema — there's no validated mapping from physical units to these ratios
anywhere in Pub 4/5 (see `docs/agent_implementation_plan.md`'s "Schema
update" note).

**M4 (2026-08-29)**: `src/metis/router/demo.py` — 8 curated `SCENARIOS`
(real cross-checked ratios only), `run_scenarios()` routes them through
the real `route_case_payload`. Branch coverage: 5 training-grid cases →
surrogate/"high", case10 → solver/"low", case15 → solver/"medium", one
novel point → loud `error` (no fabricated answer).
`scripts/router_demo_app.py` — Streamlit front end; deliberately skips
the M3 LLM layer so it runs with no Anthropic credentials. Streamlit is
its own `demo` extra (`pip install -e ".[demo]"`), kept out of `router`.
Outcomes pinned in `tests/integration/test_demo.py`. App boots and serves
(health check); live in-browser click-through not exercised here.

**M5 (2026-08-29)**: `docs/router_agent.md` — standalone write-up (the
problem, the validated Pub 5 OOD finding the confidence rule encodes, the
case10/case15 contrast, the rejected latent-geometry diagnostic, the
one-tool agent rationale, real vs. precomputed, explicit Pub 4/5 linkage,
the n=2 limitation). `README.md` — "Router agent" section with a Mermaid
architecture diagram that renders on GitHub, plus Layout/Status updates.
`agent_implementation_plan.md`'s stale 3-tool JSON sketch replaced with
the real single `route_case` tool. GitHub Actions CI green.
`docs/router_demo.gif` (2026-08-30) — 6-frame capture of the Streamlit
demo, recorded by driving headless Brave over the DevTools protocol
against the real checkpoint/DNS lookup; embedded at the top of the
README's Router agent section. **M5 complete.**

## Immediate priorities

**Track C/D (router portfolio module) is complete — M0-M5, all pushed,
CI green.** Non-blocking leftovers: verify `agent.py`'s live LLM round
trip once an `ANTHROPIC_API_KEY` exists
(`test_ask_end_to_end_live_llm_call` skips); eyeball the GitHub-rendered
README (Mermaid + `docs/router_demo.gif`).

**Active work now: the `METIS_detailed_next_steps.md` backlog** — turning
the Track A/B scientific core into something another researcher can
actually run. Execution order there: R1 config/path refactor → R2 case
registry → R3 validation → R4 preprocessing → R5 dataset artifacts → R6
common analysis API → R7 freeze regime-v1 benchmark, then Stage 2 (MLflow
+ autoencoder representation study + training framework + physics-aware
eval).

- **R1 — configurable data root: DONE (2026-08-30).** New `metis.config`
  (resolve order: `--data-root` > YAML `data.root` > `$METIS_DATA_ROOT` >
  helpful error); `configs/default.yaml`; the three `run_regime_*.py`
  scripts take `--data-root`/`--config`/`--output` and lost their
  hard-coded `DATA_ROOT`; `router/surrogate.py` resolves the checkpoint
  via `$METIS_ROUTER_CHECKPOINT` / `$METIS_PUB5_ROOT` / a sibling
  `pub5_neural_operators/` (no `/home/...` literal left under `src/`,
  `scripts/`, `pyproject.toml`); the `router` extra no longer carries the
  `file:///` neuralop_bench dep (install it editable separately).
  `tests/unit/test_config.py` covers the resolution order.
- **R2 — case registry: DONE (2026-08-30).** New `metis.data.registry`:
  `CaseDescriptor` (frozen; `case_id`, the three ratios, `nx/ny/nz`,
  `n_snapshots`, `raw_dir`/`processed_dir`/`slice_root`, `extra`; methods
  `has_raw`/`has_processed`/`has_slices`, `available_slices()`,
  `load()`→`DNSCase`, `load_slice(id)`→`SliceCase`, `require(raw=,
  slices=)`) and `CaseRegistry` (`from_config()` resolves the data root
  via `metis.config`; discovers cases from `processed/<id>/metadata.json`;
  `registry["case01"]`, `in`, `iter`, `len`, `descriptors()`; `KeyError`
  lists known cases, missing metadata keys raise `ValueError`).
  `metis.features.regime.case_grid_labels` now also accepts a registry.
  `metis.data.registry.REQUIRED_METADATA_KEYS` is the authoritative
  case-metadata schema. `tests/unit/test_registry.py` (11 tests) +
  smoke-checked against the real 11-case data tree.
- **R3 — data validation layer: DONE (2026-08-30).** New
  `metis.data.validation`: `ValidationReport` / `ValidationIssue` /
  `ValidationError` (`report.ok/errors/warnings`, `raise_if_failed()`,
  `summary()`) + validators `validate_dns_case`, `validate_slice_case`,
  `validate_case(descriptor, load=)`, `validate_compatibility([...])`.
  Structural (grid+ghost-cell shape, coordinate length/finiteness/
  monotonicity, `timesteps` ordered & `timesteps_done ⊆ timesteps`),
  numerical (NaN/Inf per field, physically-positive rho/T/P/mu/kappa/
  c_p/c_v/sos, `rmsf_* >= 0`, constant-field warning, slice fluctuation-
  drift + duplicate-frame warnings), and cross-case (matching grids &
  field-name sets). New `metis.testing.corrupt` supplies the deliberately
  corrupted fixtures; `tests/unit/test_validation.py` (14 tests) asserts
  each failure by its `check` slug. Smoke-checked on real case01/10/15
  and slices s3_center/s2_max_u.
- **R3 follow-up — RHEA axis order: FIXED (2026-08-30).** Smoke-testing
  surfaced that `HDF5Reader` extracted `coordinates['x']`/`['z']` along
  the wrong mesh axis (constant for real data). RHEA writes arrays as
  **`[z, y, x]`** (shape `(Nz+2, Ny+2, Nx+2)`), per the group. Fixed the
  reader's coordinate extraction, made `mock_dns.generate` build in the
  same order, and corrected the `[z,y,x]` shape check in
  `validate_dns_case` + docstrings in `hdf5_reader`/`physics`. Physics
  code needed no change — `wall_normal_profiles` averages axes 0 and 2
  (both homogeneous) and only reads `coordinates['y']` (axis 1, always
  correct), which is why the Pub 4 cross-checks always passed. Real
  case01/10/15 now validate with 0 warnings; box is ~1.27mm(x) x
  0.35mm(y=2·DELTA) x 0.42mm(z), a minimal flow unit as expected.
- **R4 — preprocessing layer: DONE (2026-08-30).** New
  `metis.data.preprocessing`: a `Transform` ABC (`fit`/`transform`/
  `fit_transform`/`inverse_transform`/`is_fitted`/`get_params`+
  `from_params`; subclasses self-register for `save`/`load` to plain
  JSON), `StandardScaler` (matches `evaluation.regime.standardize`
  exactly — population std, zero-variance column left at 0, so it's a
  drop-in), `MinMaxScaler`, and stateless field helpers
  `strip_ghost_cells` / `select_variables` / `subsample` /
  `interior_fields`. Fit is training-rows-only; `transform` never refits.
  `tests/unit/test_preprocessing.py` (14 tests) incl. an explicit
  no-leakage test (fit on train, `mean_` == train mean not combined
  mean, applying to test doesn't move it) and `save`/`load` round-trip.
  Deliberately NOT rewiring `evaluation.regime`'s global standardization
  — the doc says that's fine for the exploratory small-N task; leakage-
  safe splits land with R7/I2.
- **R5 — dataset abstraction + artifact caching: DONE (2026-08-30).** New
  `metis.data.datasets`: `FeatureDataset` (Level-1, one feature row per
  case — `X`, `case_ids`, `feature_names`, `feature_set`, `provenance`;
  `save`/`load` to `<dir>/data.npz` + `metadata.json`; `row(id)`,
  `matrix(subset)`) and `build_feature_dataset(registry, case_ids,
  feature_set, out_dir=, rebuild=)`. `feature_set` ∈ {compact, rich,
  bulk, mean_profile, rms_profile, pod} wraps the `regime.py` builders.
  Caching keys on a fingerprint of (case_ids, feature_set, sha256 of
  `features/{regime,physics,pod}.py`) — editing feature code busts the
  cache; raw-DNS edits are NOT detected (documented; use `--rebuild`).
  Provenance also records git commit, data_root, timestamp, shape.
  `scripts/build_feature_dataset.py` CLI; `artifacts/` gitignored.
  `tests/unit/test_feature_dataset.py` (8 tests). Real-data smoke: first
  build 11×14 in 7.4s, cached re-run 0.001s, identical X — the
  acceptance criterion (no raw re-read when unchanged).
- **R6 — standard analysis API: DONE (2026-08-30).** New `metis.analysis`
  (single module, function-dispatch — not a class hierarchy):
  `run_analysis(registry, analysis, case, *, validate=True, **options)`
  over `ANALYSES = {physics, spectra, pod, regime_features}`, each a thin
  runner adapting an existing trusted function. One `AnalysisResult`
  (analysis, case_ids, resolved config, JSON-safe outputs, `arrays`,
  validation summary, provenance) with `save`/`load`
  (outputs.json + arrays.npz) and `summary()`. Validation runs by default
  (`validate_dns_case` / `validate_slice_case` / `validate_case`).
  `scripts/analyze.py` CLI (`analyze <kind> <case> [--slice --field
  --axis --feature-set --energy-threshold --out --no-validate]`).
  `metis.config.git_commit()` factored out (also used by R5).
  `tests/unit/test_analysis.py` (11 tests). Real case01 smoke: pod
  energy_captured = 0.9900003143 (exact Pub 4 match), spectra Parseval
  rel-err 4e-11, physics/regime clean.
- **R7 — freeze regime discovery as `regime-v1` benchmark: DONE
  (2026-08-30). Stage 1 of the backlog is complete.**
  `configs/benchmarks/regime_discovery_v1.yaml` pins the spec (train
  case01-09 / OOD case10,15; `bulk` = pressure diagnostic, `rms_profile`
  = thermal diagnostic; MFA `bulk`+`rms_profile` sanity check; every
  §3/§4 metric with a 1e-6 tolerance). `metis.evaluation.benchmark`:
  `run_regime_v1(block_matrices, labels, config) -> BenchmarkResult`
  (per-block eval + MFA/naive combos + 7 frozen `checks`;
  `assert_passes()`), `run_regime_v1_from_registry` (builds the blocks
  via R5 caching). `scripts/benchmark_regime_v1.py` — the `metis
  benchmark regime-v1` command; writes `results/regime_v1.json`, exits
  non-zero on regression; reproduces §3/§4 exactly on real data (7/7
  pass). `tests/regression/test_regime_v1_benchmark.py` (5 tests) runs
  on a 60 KB committed feature artifact
  (`tests/data/regime_v1/block_features.npz`) — no DNS, CI-safe.
  FINDINGS.md §5 records the freeze. The three older
  `scripts/run_regime_*.py` still work and still reproduce their JSONs
  bit-for-bit (verified); they're kept as the exploratory trail.
- **I1 — MLflow experiment tracking: DONE (2026-08-30).** New
  `metis.tracking`: `run(experiment, *, params=, tags=, tracking_dir=,
  enabled=)` context manager over MLflow. Local-first — runs to
  `<repo>/mlruns/` unless `$MLFLOW_TRACKING_URI` is set, no server/DB
  (sets `MLFLOW_ALLOW_FILE_STORE=true`, since MLflow ≥3 gates the file
  store — `mlflow ui` needs the same flag exported). Auto-logs git
  commit, `metis.__version__`, python/platform, the flattened config,
  and `runtime_seconds` on exit; handle exposes `log_metrics` /
  `log_params` / `log_artifact` / `log_dict` / `set_tags`. **Graceful
  degradation**: no `ml` extra → `is_available()` is False and `run()`
  yields a no-op `_NullRun` (callers never branch). First consumer:
  `scripts/benchmark_regime_v1.py` logs every regime-v1 run (params =
  benchmark config + data_root; metrics = all block/combined ARI/LOCO +
  per-check pass; artifacts = `regime_v1.json` + the config yaml);
  `--no-track` disables. `metis.__init__` now sets `__version__` via
  `importlib.metadata`. `tests/unit/test_tracking.py` (6 tests,
  `importorskip mlflow`) incl. the §13 acceptance check (run id → data,
  config, metrics, artifacts) and failed-run recording. Benchmark check
  names lost their `[]` (MLflow metric-name charset): `best_block[x]` →
  `best_block__x`; `results/regime_v1.json` regenerated.
- **I2-A / I3 / I4 — autoencoder representation study: DONE, STOP
  CRITERION MET (2026-08-30).** New `metis.models` (`RepresentationModel`
  = the R4 `Transform` contract; `PCARepresentation` numpy baseline;
  `Autoencoder` small MLP, torch-gated), `metis.training.Trainer`
  (full-batch Adam + early stop + best-state + optional checkpoint +
  optional `metis.tracking` run; reproducible per seed), and
  `metis.evaluation.representation` (`evaluate_representation` — same
  ARI/LOCO/OOD-nearest-centroid instruments as regime, on any latent;
  `compare_to_baseline` — the mechanical stop-criterion check).
  `scripts/representation_study.py` (I2-A, MLflow `representation-v1`,
  `results/representation_study.json`) ran on real data: on `bulk` and
  `rms_profile`, latent dim 2, 5 seeds — **the AE latent reproduces the
  PCA subspace** (rms_profile metrics identical to 3 dp; bulk slightly
  worse; zero seed variance). `beats_baseline` False on every block.
  Recorded in **FINDINGS.md §6**; case-level representation learning is
  parked per §14. I2-B (2D slice fields, N≫9) is a deliberate decision,
  not an automatic next step. Removed the six empty `models/*/`
  scaffold subdirs. First real consumers of I1 tracking: this study +
  the benchmark. `metis.models`/`training`/`evaluation.representation`
  are reusable if I2-B is taken up.
- **I5 — physics-aware evaluation API: DONE (2026-08-30).**
  `metis.evaluation.metrics` — `pointwise_metrics` (MAE / RMSE / nRMSE /
  relative-L2 / R²; nRMSE & rel-L2 scale-invariant).
  `metis.evaluation.physical` — `physical_field_report(true, pred, *,
  profile_axis, spectrum_axis=)` bundling pointwise + mean/RMS-profile
  relative-L2 + wavenumber-spectrum agreement (rel-L2, log-spectrum
  correlation, premultiplied-peak shift); `pod_energy_agreement` on
  energy-fraction arrays. `metis.evaluation.representation` gained
  `latent_stability` (Procrustes-aligned rel-L2 across repeated fits) and
  `latent_physical_correlation` (max |corr| of any latent axis with a
  physical variable). `metis.evaluation.assessment.assess_model` — the
  §17 gate: signed deltas per metric, `is_better` only if ≥1 ML metric
  improves AND no physical diagnostic regresses. `tests/unit/{test_metrics,
  test_physical_eval,test_assessment}.py` + `test_representation_eval`
  extended (20 new tests, all numpy/scipy → run in CI). Wired into
  `scripts/representation_study.py`: latent_stability ≈ 1e-16 and
  matching latent↔physical correlations further confirm the FINDINGS §6
  stop — the AE just finds PCA's subspace.
- **I6 — reporting layer: DONE (2026-08-30). Stage 2 (I1-I6) complete.**
  `metis.reporting.Report` — accumulates markdown sections (+ GFM tables),
  a flat metrics dict, and named figures, then `write`s
  `<out>/{summary.md, metrics.json, figures/*.png}` with a provenance
  header (generated_at, metis version, git commit). Figures need the new
  `report` extra (matplotlib, Agg); without it the text+JSON still write
  and `summary.md` says the figure was skipped.
  `metis.reporting.generators` — `regime_v1_report`,
  `representation_study_report` (from the `results/*.json` shapes),
  `physics_report` (from an `AnalysisResult`). `scripts/report.py`:
  `report <kind> [--from JSON | --run-id <mlflow-id>] [--out]`, plus
  `report physics <case> --data-root ...`. `reports/` gitignored.
  `tests/unit/test_reporting.py` (5 tests, text/JSON path runs in CI;
  figure assertions guard on `has_matplotlib()`). Smoke-tested all three
  report kinds on real artifacts + the `--run-id` MLflow-artifact path.
- **I7 — user-facing `metis` CLI: DONE (2026-08-30).** `metis.cli`
  (argparse subcommands, `[project.scripts] metis = "metis.cli:main"` —
  installed by `[dev]`, no extra needed): `metis cases list`,
  `metis case validate <id> [--slices]`, `metis analyze <kind> <case>
  [...]`, `metis dataset build --feature-set --out [--cases --rebuild]`,
  `metis benchmark regime-v1 [--no-track]`, `metis report <kind>
  [--from|--run-id]` / `metis report physics <case>`, `metis --version`.
  Every subcommand wraps a `metis.*` call; `KeyError`/`RuntimeError`/
  `FileNotFoundError` become `metis: error: ...` on stderr + exit 2.
  `scripts/{analyze,report,benchmark_regime_v1,build_feature_dataset}.py`
  are now 11-line shims that forward argv to the CLI (the regime
  exploratory scripts + `representation_study.py` stay standalone).
  README quickstart rewritten around `metis ...`.
  `tests/unit/test_cli.py` (9 tests, CI-safe — mock_dns raw HDF5 only).
- **I8 — artifact (dataset/model) registry: DONE (2026-08-30).**
  `metis.registry` (distinct from `metis.data.registry`, the DNS *case*
  registry): `ArtifactRegistry("artifacts")` over one `registry.json` —
  `register(id, kind, *, status, run_id, dataset_id, path, scope,
  metrics)` (raises on duplicate id), `update`, `set_status` /
  `promote` / `deprecate`, `list(kind=, status=)`, `[id]`. `kind` ∈
  {dataset, model, report}, `status` ∈ {experimental, validated,
  deprecated}; **`promote` to `validated` requires a documented
  `scope`**. Auto-fills created/updated_at, git_commit, metis_version;
  atomic save. CLI: `metis registry add|list|show|promote|deprecate|
  set-status`, and `metis dataset build --register <id> --scope ...`
  records the built dataset with its fingerprint/shape as metrics.
  `tests/unit/test_artifact_registry.py` (9) + 3 CLI tests.
- **Docs cleanup: DONE (2026-08-30).** New `docs/README.md` (index) +
  six student-facing guides: `getting_started.md` (install → mock data →
  tests → first analysis → where to add code), `architecture.md` (the
  layer pipeline + full `src/metis/` module map + design rules),
  `data_layout.md` (data root, `metadata.json` schema, the `[z,y,x]`
  axis convention, `CaseRegistry` usage), `adding_an_analysis.md`,
  `adding_a_model.md`, `reproducibility.md`. README trimmed from 149
  lines to what/what-it-does/install/one-example/router-pointer/
  where-to-read-more; the stale Layout tree (deleted `models/*` subdirs)
  and Status section are gone — module map now lives in
  `docs/architecture.md`. The router docs
  (`docs/router_agent.md`, `docs/agent_implementation_plan.md`) stayed
  put (11 inbound refs from source docstrings). All relative links
  verified.
- **Stage 3 tail + P1/P3 packaging: DONE (2026-08-30).**
  `tests/integration/test_pipeline.py` — the full chain on synthetic data
  (ingest → validate → cached feature dataset → train-only scaler → fit a
  model → evaluate (ML + representation + `assess_model` gate) →
  `AnalysisResult` save/reload → `physical_field_report` → `Report`
  write); a second test covers the torch leg (`Autoencoder` + `Trainer` +
  `metis.tracking`, `importorskip`). `tests/regression/
  test_physics_regression.py` (15) — pins the 11-case `bulk` block (from
  the committed `block_features.npz`) against `tests/data/
  physics_regression/bulk_reference.json`, the values published in Pub 4's
  `case_setup_table.json`; bit-for-bit at `rel=1e-9`, no DNS needed.
  CI (`.github/workflows/ci.yml`) gained a **`package` job**: `python -m
  build` → install the wheel in a clean venv → `metis --version` /
  `--help` / `cases list` smoke. `pyproject.toml` polished — `readme`,
  `authors`, `keywords`, `classifiers`, `[project.urls]`. Wheel builds
  and the `metis` entry point works from it. Full suite 249 passed / 1
  skipped; clean-venv 213 / 8.
- **Plan superseded (2026-08-30).** `METIS_detailed_next_steps_updated.md`
  replaces the R/I/P milestone plan. It rates the platform "early but
  credible ML experimentation platform", says **stop broad architecture
  refactoring** (§23), and reprioritises: correctness debt → richer ML
  research → an industry temporal track → one model to deployment.
- **Phase A (correctness / small debt, updated-plan §18-22): DONE
  (2026-08-30).**
  - **§18 AE seed bug (P0):** `_make_net` reset torch to seed 0 inside
    net construction → every "seed" started from identical weights.
    Fixed (seed drives init); reran I2-A. Headline result survives (AE
    beats PCA on no block) but the reasoning changed: on `rms_profile`
    the AE is now *worse on average and highly seed-sensitive* (ARI vs
    Thw_Tc 0.24 ± 0.23 vs PCA 0.48). FINDINGS §6 rewritten with a
    correction note; `results/representation_study.json` + MLflow
    regenerated.
  - **§19 persistent benchmark cache:** `metis benchmark regime-v1`
    caches each feature block under `artifacts/datasets/regime-v1/<block>/`
    (`--cache-dir` / `--no-cache` / `--rebuild`; `provenance["cache"]`
    reports reused/built). Re-run 8.5s → 0.9s.
  - **§20 stronger fingerprints:** the dataset fingerprint now also
    covers a per-file **source manifest** (raw/processed/slice: path,
    size, mtime, + metadata content hash) and the ingestion +
    preprocessing source — touching a raw `.h5` or editing a
    `metadata.json` busts the cache.
  - **§21 `from_config` mapping:** `resolve_data_root` takes a `config`
    mapping; order is explicit arg > mapping > file > env.
  - **§22 strict validation:** `run_analysis(strict=True)` default — a
    validation ERROR aborts (`ValidationError`); `strict=False` /
    `metis analyze --allow-invalid` proceeds and stamps
    `provenance["validation_overridden"]`.
- **Phase B — I2-B: protocol DRAFTED (2026-08-30), awaiting freeze (B1).**
  `docs/i2b_representation_protocol.md` + `configs/experiments/
  i2b_representation_v1.yaml`. Question: does a conv-AE latent from
  centre-plane `u'` snapshots (`s3_center`, field `u`, 96×96, ~500/case)
  capture structure PCA/POD misses, at matched latent dim? Splits:
  train = case01-09 snapshots 0-399; val = same cases 400-499
  (chronological); OOD = all of case10/case15 (off-grid thermal). +
  secondary Pb_Pc=5.0 pressure-holdout. Baseline PCA(=snapshot POD) at
  k∈{2,4,8,16,32}; candidate = small fixed conv-AE (16/32/64 stride-2,
  GELU, MSE) at the same k; headline k = where PCA first hits ≥90%
  reconstructed variance. Eval: reconstruction + seed robustness +
  `latent_stability` + OOD degradation ratio + physical fidelity
  (`profile`/`spectrum`/`pod_energy` agreement) + `latent_physical_
  correlation`, compared via `assess_model`. Accept = ≥3/5 seeds
  `is_better` beyond seed noise; else record the negative in FINDINGS §7
  and stop (no VAE/U-Net/operator escalation). Execution: B1 freeze →
  B2 Level-2 slice-dataset builder → B3 baseline + pick k → B4 mini-batch
  `Trainer` + conv-AE → B5 train all seeds → B6-B7 eval + decision gate →
  B8-B9 writeup/registry.
  - **B1 — protocol FROZEN (2026-08-30).** `docs/i2b_representation_
    protocol.md` + `configs/experiments/i2b_representation_v1.yaml`; user
    confirmed all four freeze decisions.
  - **B2 — Level-2 slice-dataset builder: DONE (2026-08-30).** New
    `metis.data.datasets.slice_dataset`: `SliceDataset` (raw `(n,NX,NZ)`
    snapshots + `case_ids` + `snapshot_idx` + a train-only global
    `scaler`; `standardize()` / `inverse()`; `save`/`load` as
    `{train,val,ood}.npz` + `metadata.json`) and `build_slice_dataset(
    registry, field, slice_id, splits, out_dir=, rebuild=)` with a
    source-manifest fingerprint + cache (touching a slice `.npy` busts
    it). `normalize_split_config` maps the yaml `splits.<name>` shape.
    `metis dataset build-slices --experiment-config ... --split primary`.
    `tests/unit/test_slice_dataset.py` (7). Real-data run of the primary
    split: **train 3600 / val 900 / ood 1000** 96×96 samples, ~170 MB
    artifact, scaler mean≈0 std 0.073.
  - **B3 — PCA/POD baseline: DONE (2026-08-30).**
    `metis.evaluation.representation.{explained_variance_curve,
    choose_latent_dim}` + `scripts/i2b_baseline.py`
    (`results/i2b_baseline.json`, MLflow `i2b-representation/baseline-pca`).
    PCA ≡ method-of-snapshots POD to **2e-15**. Linear reconstruction is
    **high-rank**: k=32 captures only **77.6%** of training variance →
    the ≥90% rule has no answer in the grid; `headline_latent_dim` frozen
    to **32** (largest grid dim, `reached=False` noted). val relL2
    0.94→0.65 over k=2→32; OOD relL2 0.53→0.35 (OOD reconstructs *better*
    than held-out train snapshots — noted). FINDINGS.md §7 started.
  - **B4 — mini-batch Trainer + conv-AE: DONE (2026-08-30).**
    `Trainer` gained `batch_size` (shuffled mini-batch loop, full-batch
    by default / back-compatible) and an `X_val` arg to `fit` that
    switches early stopping to validation MSE; history reports
    `best_<monitor>_mse` plus a stable `best_train_mse`.
    `metis.models.conv_autoencoder.ConvAutoencoder` (torch-gated,
    imported directly): 3× stride-2 conv encoder (H→H/8) + linear
    bottleneck + mirrored ConvTranspose decoder per protocol §4, operates
    on `(n, H, W)`, seed drives weight init, `save`/`load` via the
    `Transform` registry. `tests/unit/test_conv_autoencoder.py` (12) +
    4 new `Trainer` tests. Clean-venv CI sim green (torch-gated tests
    skip). CI green (82fa52d).
- **Next: B5** — training script: 5 seeds × k grid {2,4,8,16,32} on the
  frozen slice dataset
  (`artifacts/datasets/i2b_representation_v1_primary_u_s3_center`),
  MLflow-logged, best-state checkpoints. Then B6-B7 (eval + `assess_model`
  decision gate) → B8-B9 (FINDINGS §7 writeup, register if positive,
  `metis report`). After Phase B: Phase C (virtual-sensor temporal ML),
  D (SQL layer), E (one model → FastAPI → Docker → CI/CD → cloud →
  monitoring).

Out of scope until a deliberate decision (both docs agree): live
HPC/Slurm solver connection, physical-units → `(Pb_Pc, Thw_Tc, Tcw_Tc)`
mapping, more OOD cases for the n=2 router rule, and the
Kubernetes/Kafka/Spark/Airflow/Terraform tier.

## Constraints to respect

- 2-4h/day + long weekend sessions.
- Personal job-search deadline: Jan 2027 — the personal module (Track C/D)
  must ship on its own timeline, not wait on Track A/B being "complete."
- ERC PoC (lead PI, 7-person team, budget/procurement) is the primary
  day-job commitment — this project runs in parallel, not instead of it.
- Don't let scope creep past what `docs/roadmap_v2_analysis_framework.md`
  §38 marks "do now" without a deliberate decision to move the line.

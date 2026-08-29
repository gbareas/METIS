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

**Track C/D, M0-M4 (2026-08-29)**: `src/metis/router/` is a working
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

## Immediate priorities

M0-M4 are done, committed, and pushed. GitHub Actions CI is green as of
2026-08-29 (it had failed on every run since the initial commit — two
latent bugs: ruff isort not resolving `metis` as first-party in a clean
checkout, and `.gitignore`'s unanchored `data/` rule silently excluding
the whole `src/metis/data/` package; both fixed, see git history).

1. Track C/D, M5 (packaging) — the active next milestone: README with
   architecture diagram, short write-up linking the module to Pub 4/5,
   demo GIF/video, clean repo structure.
2. Track C/D, M3 follow-up (blocked, not actionable here): verify
   `agent.py`'s live LLM round trip once Anthropic API credentials are
   available in this environment
   (`tests/integration/test_agent.py::test_ask_end_to_end_live_llm_call`
   currently skips).
3. Track A/B: no active priority — regime discovery reached a settled,
   documented stopping point (`FINDINGS.md` §1-4). Defer
   MLflow/baselines/advanced ML/deployment (roadmap v2 §38 "do now" list)
   until there's a specific reason to pick it back up.

## Constraints to respect

- 2-4h/day + long weekend sessions.
- Personal job-search deadline: Jan 2027 — the personal module (Track C/D)
  must ship on its own timeline, not wait on Track A/B being "complete."
- ERC PoC (lead PI, 7-person team, budget/procurement) is the primary
  day-job commitment — this project runs in parallel, not instead of it.
- Don't let scope creep past what `docs/roadmap_v2_analysis_framework.md`
  §38 marks "do now" without a deliberate decision to move the line.

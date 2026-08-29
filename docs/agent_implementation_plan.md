# Hybrid Solver-Orchestration Agent — Implementation Plan

## Essence

An agent that routes between a fast neural-operator surrogate and a full physics
solver for transcritical channel-flow queries, deciding which to trust based on
an out-of-distribution (OOD) confidence signal, and explaining that decision in
plain language.

**Why it needs to exist:** neural operators are fast but fail silently outside
their training distribution — especially near the pseudo-critical point in
transcritical flows, where fluid properties diverge sharply. Today the only
options are "trust the surrogate blindly" (risky) or "always run the full
solver" (throws away the speed gain). This agent productizes a validated
confidence policy instead of introducing a new unvalidated one — it turns the
OOD/LOO findings from Pub 5 into something that behaves.

**The claim being made:** the router's logic — *why* a given confidence score
triggers fallback, and how that maps to the Pub 5 findings — is the scientific
core of this project and must be defensible without reference to any
AI-assisted tooling. Everything downstream (LLM orchestration, tool wrappers,
UI) is standard 2026 engineering and can be built with current SOTA tools
(e.g. Claude Code) without caveat.

---

## Architecture

### Core router (no LLM) — build and validate first

```python
def route(case_params: dict) -> dict:
    prediction = surrogate_infer(case_params)
    confidence = confidence_score(case_params)   # reuses Pub 5 OOD diagnostic

    if confidence >= THRESHOLD:
        return {
            "source": "surrogate",
            "result": prediction,
            "confidence": confidence,
        }
    else:
        result = solver_lookup(case_params)       # precomputed fallback, see M2
        return {
            "source": "full_solver",
            "result": result,
            "confidence": confidence,
        }
```

### Agent layer — wraps the router as tools for an LLM orchestrator

**Schema update (M1, 2026-08-29):** the `inlet_pressure`/`inlet_temperature`/
`mass_flow_rate`/`geometry_id` schema below is superseded. There's no
validated mapping anywhere in Pub 4/5 from continuous physical units to the
`(Pb_Pc, Thw_Tc, Tcw_Tc)` ratios the surrogate and confidence diagnostic
actually operate on — inventing one would be exactly the kind of new
unvalidated claim this project is meant to avoid. `route()`/`confidence_score`/
`surrogate_infer` (`src/metis/router/`) take `{"Pb_Pc": float, "Thw_Tc": float,
"Tcw_Tc": float}` directly instead. Update this JSON block when M3 (agent
layer) is actually built, so it matches `src/metis/router/` rather than the
original aspiration.

```json
{
  "tools": [
    {
      "name": "surrogate_infer",
      "description": "Fast neural-operator prediction for a transcritical channel-flow case.",
      "input_schema": {
        "type": "object",
        "properties": {
          "inlet_pressure": {"type": "number"},
          "inlet_temperature": {"type": "number"},
          "mass_flow_rate": {"type": "number"},
          "geometry_id": {"type": "string"}
        },
        "required": ["inlet_pressure", "inlet_temperature", "mass_flow_rate", "geometry_id"]
      }
    },
    {
      "name": "confidence_score",
      "description": "OOD/confidence diagnostic for a case, reusing the Pub 5 LOO/OOD methodology.",
      "input_schema": {
        "type": "object",
        "properties": {
          "inlet_pressure": {"type": "number"},
          "inlet_temperature": {"type": "number"},
          "mass_flow_rate": {"type": "number"},
          "geometry_id": {"type": "string"}
        },
        "required": ["inlet_pressure", "inlet_temperature", "mass_flow_rate", "geometry_id"]
      }
    },
    {
      "name": "solver_lookup",
      "description": "Retrieve a precomputed full-solver (RHEA) result for a case, used as fallback when confidence is low.",
      "input_schema": {
        "type": "object",
        "properties": {
          "inlet_pressure": {"type": "number"},
          "inlet_temperature": {"type": "number"},
          "mass_flow_rate": {"type": "number"},
          "geometry_id": {"type": "string"}
        },
        "required": ["inlet_pressure", "inlet_temperature", "mass_flow_rate", "geometry_id"]
      }
    }
  ]
}
```

The LLM orchestrator's job is narrow: parse a natural-language query into the
structured params above, call the tools in order, and turn the router's output
into a short plain-language explanation ("used the surrogate — confidence
0.91, well inside training range" / "routed to full solver — case sits near
the pseudo-critical point where the surrogate is known to degrade").

---

## Milestones

- [x] **M0 — Scoping.** Freeze the case family (Pub 4 transcritical channel-flow
  domain). Confirm access to an existing trained surrogate checkpoint. Define
  "full-solver answer" as precomputed data from Pub 4/5 — no live HPC needed
  for the MVP. *(2026-08-29: checkpoint confirmed at
  `pub5_neural_operators/runs/campaignB/A2/xy_slice_1/unet_raw_ood/best.pt`.)*
- [x] **M1 — Core router (no LLM).** Implement `route()`. Validate against the
  Pub 5 LOO/OOD campaigns: does it correctly flag the cases already shown to
  break the surrogate? **This is the load-bearing milestone** — everything
  else depends on it being right and being something you can explain cold.
  *(2026-08-29: `src/metis/router/` — `confidence_score` is a rule-based
  regime-envelope/`Tcw_Tc` check per FINDINGS §5.13 (the latent-geometry
  diagnostic in §5.17 was tried and failed, and is deliberately not used);
  `surrogate_infer` wraps the frozen U-Net checkpoint and reproduces its
  recorded blind-OOD errors for case10/case15 to ~3e-4 relative precision;
  `route()` correctly trusts the surrogate for case01 (in-envelope) and
  falls back to the full solver for case10. See
  `tests/integration/test_router.py`.)*
- [x] **M2 — Fallback path.** Build the precomputed lookup for `solver_lookup`.
  Document clearly (in code and README) that it's precomputed, not a live
  solve — don't let it read as live. *(2026-08-29: `src/metis/router/
  solver.py` — matches `(Pb_Pc, Thw_Tc, Tcw_Tc)` against
  `case_descriptors.json` (near-exact match, not nearest-neighbor — a
  genuinely novel point raises `KeyError`, it's never fabricated) and
  returns the group's actual converged DNS RMS fields for that case, in
  the same `{"fields", "field_means"}` shape as `surrogate_infer` so the
  two are directly comparable. Covers case01-09/case10/case15 only —
  11 simulated cases, not 15 as M1's docstring imprecisely said.)*
- [x] **M3 — Agent layer.** Wire the three tools to an LLM orchestrator
  (Claude, tool use). Natural-language query in, routed result + explanation
  out. *(2026-08-29: `src/metis/router/agent.py` — deliberately **one**
  tool, `route_case`, not three. Letting the LLM sequence
  `surrogate_infer`/`confidence_score`/`solver_lookup` itself would mean
  the LLM enacts the routing decision — exactly the "new unvalidated
  policy on top of the validated one" this project's own design principle
  rules out. `route_case` wraps `route()` unmodified; the LLM only parses
  the query and explains the (already-decided) result. Model:
  `claude-opus-5`. Built on the SDK's tool runner (beta), not a
  hand-written loop — single tool, single turn, no need for the runner's
  per-turn hooks. `_route_case`'s logic is validated against real
  cases/checkpoint in `tests/integration/test_agent.py`; the actual live
  LLM round trip (`ask()`) is untested in this environment — no
  `ANTHROPIC_API_KEY` or `ant auth login` profile configured here, so
  that test skips. Needs live credentials to verify before relying on
  it.)*
- [x] **M4 — Interface + demo.** Minimal Streamlit or CLI front end. 5–10
  curated scenarios that clearly exercise both branches (confident surrogate
  case, routed-to-solver case). *(2026-08-29: `src/metis/router/demo.py` —
  8 curated `SCENARIOS`, every ratio a real cross-checked value (case01-09/
  10/15 descriptors + the novel point already used as a router test case),
  nothing invented for the demo. Coverage: 5 training-grid cases → surrogate/
  "high", case10 → solver/"low", case15 → solver/"medium", novel point →
  loud `error` (no fabricated answer). `run_scenarios()` routes them through
  the real `route_case_payload` (frozen checkpoint + confidence diagnostic +
  precomputed DNS lookup). `scripts/router_demo_app.py` — Streamlit front
  end over `run_scenarios()`; deliberately does **not** call the M3 LLM
  layer, so the demo runs without Anthropic credentials. Streamlit lives in
  its own `demo` extra (`pip install -e ".[demo]"`), kept out of `router` so
  the router's own tests don't pull in a UI framework. Outcomes pinned in
  `tests/integration/test_demo.py` (10 tests, skip without the checkpoint,
  same as `test_router.py`). Verified the app boots and serves
  (`/_stcore/health` → ok); the live in-browser click-through is untested
  here.)*
- [ ] **M5 — Packaging.** README with architecture diagram, short write-up
  explicitly linking the project to Pub 4/5, demo video or GIF, clean repo
  structure.

---

## Definition of Done

- Runs reliably end-to-end on a fixed demo set — same input, same route,
  every time.
- Router decisions match what Pub 5 already established — this productizes
  validated science, it doesn't introduce new unvalidated claims.
- A recruiter or interviewer can understand the project from the README in
  under 2 minutes, without running any code.
- You can walk through M1's confidence logic unaided, live, in an interview.

## Explicitly Out of Scope (v1)

- New physics regimes or geometries beyond the frozen case family
- Retraining or fine-tuning the surrogate
- A general-purpose agent framework beyond this one use case
- A live HPC/Slurm solver connection (future work item, not required for v1)

---

## Timeline (target: ~4 weeks, 2–4h weekdays + long weekend sessions)

| Week | Focus |
|---|---|
| 1 | M0 + M1 core loop, basic routing working end-to-end |
| 2 | M1 validated against Pub 5 OOD/LOO cases (the technical core) |
| Weekend | M3 explanation generation + minimal UI |
| 3 | M2 fallback wiring, logging/traces, test cases across geometries |
| 4 / final weekend | M4 demo scenarios, M5 README + architecture diagram + write-up |

If the timeline is at risk, cut demo polish before cutting M1 validation.

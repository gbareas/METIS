"""Curated demo scenarios for the router (Track C/D milestone M4).

Every ratio below is a real, already-cross-checked value — case01-09/10/15
from `pub4_grassmann_rom/results/case_descriptors.json` (see
`tests/integration/test_router.py`), plus one deliberately novel point
(also already used as a test case) — nothing here is invented for the
demo. The 8 scenarios exercise every branch `route()` can take:

    - 5 training-grid cases (case01, 03, 05, 07, 09 — the 3x3 grid's
      corners + center): confidence "high", routed to the surrogate.
    - case10: `Tcw_Tc` crosses the pseudo-critical boundary, confidence
      "low", routed to the precomputed solver (this is the case the
      surrogate is known to collapse on).
    - case15: a milder thermal excursion beyond the training envelope,
      confidence "medium", also routed to the precomputed solver.
    - one genuinely novel operating point: routed to the precomputed
      solver like the two above, but no DNS case matches it, so it fails
      loudly (an "error" entry) instead of fabricating an answer — the
      demo's honesty check, not just its happy path.
"""
from __future__ import annotations

from dataclasses import dataclass

from metis.router.agent import route_case_payload


@dataclass
class Scenario:
    name: str
    description: str
    Pb_Pc: float
    Thw_Tc: float
    Tcw_Tc: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("case01", "Training grid: lowest pressure, mildest wall heating", 1.5, 1.1, 0.95),
    Scenario("case03", "Training grid: lowest pressure, strongest wall heating", 1.5, 1.4, 0.80),
    Scenario("case05", "Training grid center", 2.0, 1.2, 0.90),
    Scenario("case07", "Training grid: highest pressure, mildest wall heating", 5.0, 1.1, 0.95),
    Scenario("case09", "Training grid: highest pressure, strongest wall heating", 5.0, 1.4, 0.80),
    Scenario(
        "case10 (OOD)",
        "Cold wall crosses into supercritical — the surrogate's known collapse case",
        1.5, 1.185, 1.035,
    ),
    Scenario(
        "case15 (OOD)",
        "Mild thermal excursion beyond the training envelope",
        1.5, 1.132, 0.982,
    ),
    Scenario(
        "novel point",
        "Untested operating point — no DNS case in the database matches it",
        2.5, 1.15, 1.05,
    ),
)


def run_scenarios(scenarios: tuple[Scenario, ...] = SCENARIOS) -> list[dict]:
    """Route every scenario through the real router and return one result
    dict per scenario (each also carries its `Scenario` under `"scenario"`)."""
    results = []
    for scenario in scenarios:
        payload = route_case_payload(scenario.Pb_Pc, scenario.Thw_Tc, scenario.Tcw_Tc)
        results.append({"scenario": scenario, **payload})
    return results

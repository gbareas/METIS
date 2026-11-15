"""Core OOD-gated router (Track C/D milestones M1+M2, no LLM yet — see
docs/agent_implementation_plan.md).

`route()` decides between the fast neural-operator surrogate
(`metis.router.surrogate.surrogate_infer`) and the precomputed full-solver
answer (`metis.router.solver.solver_lookup`) based on the validated
confidence diagnostic (`metis.router.confidence.confidence_score`). Only
`"high"` confidence is trusted with the surrogate — see `confidence.py`'s
docstring for why the policy is this conservative given only n=2 OOD
cases back it.
"""
from __future__ import annotations

from metis.router.confidence import confidence_score
from metis.router.solver import solver_lookup
from metis.router.surrogate import surrogate_infer


def route(case_params: dict) -> dict:
    """Route one query to the surrogate or the full solver.

    `case_params`: {"Pb_Pc": float, "Thw_Tc": float, "Tcw_Tc": float}.
    `Tcw_Tc` is required for the confidence check even though the
    surrogate itself is only conditioned on `(Pb_Pc, Thw_Tc)` — see
    `metis.router.confidence` for why.

    Falling back to the full solver only succeeds if `case_params`
    matches one of the group's 11 already-simulated cases — see
    `metis.router.solver` for why that's a deliberate limitation, not a
    bug: there's no precomputed answer for a genuinely novel point.
    """
    confidence = confidence_score(case_params)

    if confidence.trusts_surrogate():
        prediction = surrogate_infer(case_params)
        return {"source": "surrogate", "result": prediction, "confidence": confidence}

    result = solver_lookup(case_params)
    return {"source": "full_solver", "result": result, "confidence": confidence}

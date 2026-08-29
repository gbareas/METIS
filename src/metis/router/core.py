"""Core OOD-gated router (Track C/D milestone M1, no LLM yet — see
docs/agent_implementation_plan.md).

`route()` decides between the fast neural-operator surrogate
(`metis.router.surrogate.surrogate_infer`) and the full DNS solver based
on the validated confidence diagnostic
(`metis.router.confidence.confidence_score`). Only `"high"` confidence is
trusted with the surrogate — see `confidence.py`'s docstring for why the
policy is this conservative given only n=2 OOD cases back it.
"""
from __future__ import annotations

from metis.router.confidence import confidence_score
from metis.router.surrogate import surrogate_infer


def solver_lookup(case_params: dict) -> dict:
    """Precomputed full-solver fallback — milestone M2, not yet built.

    This must stay a precomputed lookup against the group's existing DNS
    database, never a live solve (see agent_implementation_plan.md's
    Definition of Done) — so it will only ever be able to answer for the
    15 cases that already have DNS output, not arbitrary (Pb_Pc, Thw_Tc).
    """
    raise NotImplementedError(
        "solver_lookup is milestone M2 — route() currently falls back to "
        "this for any non-'high'-confidence case, but there's no "
        "precomputed lookup wired up yet."
    )


def route(case_params: dict) -> dict:
    """Route one query to the surrogate or the full solver.

    `case_params`: {"Pb_Pc": float, "Thw_Tc": float, "Tcw_Tc": float}.
    `Tcw_Tc` is required for the confidence check even though the
    surrogate itself is only conditioned on `(Pb_Pc, Thw_Tc)` — see
    `metis.router.confidence` for why.
    """
    confidence = confidence_score(case_params)

    if confidence.trusts_surrogate():
        prediction = surrogate_infer(case_params)
        return {"source": "surrogate", "result": prediction, "confidence": confidence}

    result = solver_lookup(case_params)
    return {"source": "full_solver", "result": result, "confidence": confidence}

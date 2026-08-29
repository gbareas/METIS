"""LLM agent layer (Track C/D milestone M3, docs/agent_implementation_plan.md).

The LLM's job is deliberately narrow: parse a natural-language query into
`(Pb_Pc, Thw_Tc, Tcw_Tc)`, call `route_case` — the single tool, a thin
wrapper around `metis.router.core.route()` — and turn the result into a
plain-language explanation. The LLM never decides surrogate vs. full
solver; that decision is made deterministically by `route()` before the
LLM sees the outcome. This is why there is only *one* tool here rather
than the three in the original plan doc's tool-schema sketch (
`surrogate_infer`, `confidence_score`, `solver_lookup` called
separately): letting the LLM sequence those three itself would mean the
LLM enacts the routing policy, which is exactly the "new unvalidated
policy on top of the validated one" the project's own design principle
rules out (see agent_implementation_plan.md's "The claim being made").

Uses the Anthropic SDK's tool runner (beta) rather than a hand-written
loop, since this is a single-tool, single-turn workflow with no need for
the runner's per-turn hooks.
"""
from __future__ import annotations

from anthropic import Anthropic, beta_tool

from metis.router.core import route

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are an assistant that explains routing decisions for
a transcritical channel-flow surrogate/solver system. When the user asks
about a case, call route_case with its (Pb_Pc, Thw_Tc, Tcw_Tc) ratios —
ask a clarifying question if any are missing rather than guessing.

Report the tool's output plainly: which source answered (the fast
surrogate or the full solver), the confidence level and the reason given
for it, and the predicted field means. If the tool returns an error
(no precomputed solver answer exists for that exact operating point),
say so plainly rather than presenting made-up numbers.

Never second-guess, override, or re-derive the routing decision — it is
made deterministically before you see it. Your job is to explain it, not
to re-decide it."""


def _route_case(Pb_Pc: float, Thw_Tc: float, Tcw_Tc: float) -> dict:
    """Undecorated core logic — the thing `route_case` calls, and what
    tests call directly without going through the LLM/tool-runner."""
    try:
        result = route({"Pb_Pc": Pb_Pc, "Thw_Tc": Thw_Tc, "Tcw_Tc": Tcw_Tc})
    except KeyError as e:
        return {"error": str(e)}

    confidence = result["confidence"]
    payload = {
        "source": result["source"],
        "confidence_level": confidence.level,
        "confidence_reason": confidence.reason,
        "field_means": result["result"]["field_means"],
    }
    if result["source"] == "full_solver":
        payload["case"] = result["result"]["case"]
    return payload


@beta_tool
def route_case(Pb_Pc: float, Thw_Tc: float, Tcw_Tc: float) -> dict:
    """Route a transcritical channel-flow case to the fast surrogate or
    the full solver, based on a validated confidence diagnostic, and
    return the result.

    Args:
        Pb_Pc: Bulk pressure over the CO2 critical pressure.
        Thw_Tc: Hot-wall temperature over the CO2 critical temperature.
        Tcw_Tc: Cold-wall temperature over the CO2 critical temperature.
    """
    return _route_case(Pb_Pc, Thw_Tc, Tcw_Tc)


def ask(query: str, client: Anthropic | None = None) -> str:
    """Run one natural-language query through the agent end to end and
    return its final plain-text explanation."""
    client = client or Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[route_case],
        messages=[{"role": "user", "content": query}],
    )
    final = None
    for message in runner:
        final = message
    if final is None:
        return ""
    return next((block.text for block in final.content if block.type == "text"), "")

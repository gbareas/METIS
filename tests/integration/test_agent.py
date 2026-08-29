"""Tests for the Track C/D agent layer (M3).

`_route_case` (the tool's undecorated core logic) is tested directly
against the real checkpoint/DNS data, same as the rest of the router —
no LLM call involved. `ask()` (the actual LLM round trip) is exercised
only if `ANTHROPIC_API_KEY` is set, since it costs real API credits and
needs live credentials this environment doesn't have configured.
"""
import os

import pytest

torch = pytest.importorskip("torch")
neuralop_bench = pytest.importorskip("neuralop_bench")
anthropic = pytest.importorskip("anthropic")

from neuralop_bench.data import load_descriptors

from metis.router.agent import _route_case, ask
from metis.router.surrogate import CHECKPOINT_PATH

if not CHECKPOINT_PATH.exists():
    pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}", allow_module_level=True)

DESCRIPTORS = load_descriptors()


def _case_params(case: int) -> dict:
    desc = DESCRIPTORS[f"case{case:02d}"]
    return desc["Pb_Pc"], desc["Thw_Tc"], desc["Tcw_Tc"]


def test_route_case_reports_surrogate_for_in_distribution_case():
    payload = _route_case(*_case_params(1))
    assert payload["source"] == "surrogate"
    assert payload["confidence_level"] == "high"
    assert "field_means" in payload


def test_route_case_reports_full_solver_for_case10():
    payload = _route_case(*_case_params(10))
    assert payload["source"] == "full_solver"
    assert payload["confidence_level"] == "low"
    assert payload["case"] == 10
    assert "field_means" in payload


def test_route_case_returns_error_dict_for_novel_point_instead_of_raising():
    # Tcw_Tc >= 1.0 crosses the pseudo-critical boundary, so confidence_score
    # flags this "low" and route() falls back to solver_lookup — which has
    # no case matching this exact (Pb_Pc, Thw_Tc, Tcw_Tc) combination.
    payload = _route_case(2.5, 1.15, 1.05)
    assert "error" in payload
    assert "No precomputed DNS case matches" in payload["error"]


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY set in this environment",
)
def test_ask_end_to_end_live_llm_call():
    desc = DESCRIPTORS["case01"]
    query = (
        f"I have a case with Pb_Pc={desc['Pb_Pc']}, Thw_Tc={desc['Thw_Tc']}, "
        f"Tcw_Tc={desc['Tcw_Tc']}. Which source would you use and why?"
    )
    answer = ask(query)
    assert "surrogate" in answer.lower()

"""Tests for the Track C/D demo scenarios (M4, docs/agent_implementation_plan.md).

`metis.router.demo` is just curated inputs plus a thin loop over
`route_case_payload`, so this runs against the real frozen checkpoint /
DNS lookup like the rest of the router integration tests — no LLM, no
Streamlit. It pins the routing outcome for every curated scenario so the
demo can't silently start showing a different story than M1/M2 validated.
"""
import pytest

torch = pytest.importorskip("torch")
neuralop_bench = pytest.importorskip("neuralop_bench")

from metis.router.demo import SCENARIOS, Scenario, run_scenarios
from metis.router.surrogate import CHECKPOINT_PATH

if not CHECKPOINT_PATH.exists():
    pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}", allow_module_level=True)

RESULTS = run_scenarios()
BY_NAME = {r["scenario"].name: r for r in RESULTS}


def test_every_scenario_produces_one_result_carrying_its_scenario():
    assert len(RESULTS) == len(SCENARIOS)
    for scenario, result in zip(SCENARIOS, RESULTS):
        assert result["scenario"] is scenario
        assert isinstance(result["scenario"], Scenario)


@pytest.mark.parametrize("name", ["case01", "case03", "case05", "case07", "case09"])
def test_training_grid_scenarios_route_to_the_surrogate_with_high_confidence(name):
    result = BY_NAME[name]
    assert result["source"] == "surrogate"
    assert result["confidence_level"] == "high"
    assert "field_means" in result


def test_case10_scenario_routes_to_the_precomputed_solver_at_low_confidence():
    result = BY_NAME["case10 (OOD)"]
    assert result["source"] == "full_solver"
    assert result["confidence_level"] == "low"
    assert result["case"] == 10
    assert "field_means" in result


def test_case15_scenario_routes_to_the_precomputed_solver_at_medium_confidence():
    result = BY_NAME["case15 (OOD)"]
    assert result["source"] == "full_solver"
    assert result["confidence_level"] == "medium"
    assert result["case"] == 15
    assert "field_means" in result


def test_novel_point_scenario_fails_loudly_instead_of_fabricating_an_answer():
    result = BY_NAME["novel point"]
    assert "error" in result
    assert "No precomputed DNS case matches" in result["error"]
    assert "field_means" not in result


def test_scenario_mix_exercises_both_branches_and_the_honest_failure():
    # The three counts the Streamlit front end reports back to the user.
    n_surrogate = sum(1 for r in RESULTS if r.get("source") == "surrogate")
    n_solver = sum(1 for r in RESULTS if r.get("source") == "full_solver")
    n_error = sum(1 for r in RESULTS if "error" in r)
    assert (n_surrogate, n_solver, n_error) == (5, 2, 1)

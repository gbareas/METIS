"""Integration tests for the Track C/D router (M1+M2) against the real,
frozen pub5_neural_operators checkpoint and DNS-derived RMS fields.

Skipped whenever the `router` extra isn't installed or the checkpoint
isn't reachable — these deliberately depend on this machine's local
group data/checkpoints, not anything shipped in the repo (see
metis.router.surrogate for why: the checkpoint lives in the sibling
pub5_neural_operators project, not under version control here).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
neuralop_bench = pytest.importorskip("neuralop_bench")

from neuralop_bench.data import SliceCase, load_descriptors
from neuralop_bench.metrics import rel_l2_rms

from metis.router.confidence import confidence_score
from metis.router.core import route
from metis.router.solver import solver_lookup
from metis.router.surrogate import CHECKPOINT_PATH, surrogate_infer

if not CHECKPOINT_PATH.exists():
    pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}", allow_module_level=True)

DESCRIPTORS = load_descriptors()

# Recorded blind-OOD errors, pub5_neural_operators/runs/campaignB/A2/
# xy_slice_1/unet_raw_ood/results.json — the ground truth surrogate_infer
# must reproduce.
RECORDED_REL_L2_RMS = {
    10: {"u": 0.6284053325653076, "T": 0.9221926331520081, "cp": 0.4828799068927765},
    15: {"u": 0.6499457359313965, "T": 0.36855536699295044, "cp": 0.5629278421401978},
}


def _case_params(case: int) -> dict:
    desc = DESCRIPTORS[f"case{case:02d}"]
    return {"Pb_Pc": desc["Pb_Pc"], "Thw_Tc": desc["Thw_Tc"], "Tcw_Tc": desc["Tcw_Tc"]}


@pytest.mark.parametrize("case", [10, 15])
def test_surrogate_infer_reproduces_recorded_ood_error(case):
    sc = SliceCase.load(case, "xy_slice_1", None, DESCRIPTORS)
    pred = surrogate_infer(_case_params(case))
    pred_phys = np.stack([pred["fields"][f] for f in ("u", "T", "cp")])
    errs = dict(zip(("u", "T", "cp"), (float(e) for e in rel_l2_rms(pred_phys, sc.conv_rms))))

    for field, recorded in RECORDED_REL_L2_RMS[case].items():
        assert errs[field] == pytest.approx(recorded, rel=1e-3)


def test_route_trusts_surrogate_for_in_distribution_training_case():
    result = route(_case_params(1))  # case01: fully inside the training envelope
    assert result["source"] == "surrogate"
    assert result["confidence"].level == "high"
    assert "fields" in result["result"]


def test_route_falls_back_to_precomputed_solver_for_case10():
    # confidence_score correctly flags case10 as unreliable (validated
    # against the recorded T' collapse in test_router_confidence.py); M2
    # now serves the real precomputed DNS RMS fields for it instead.
    result = route(_case_params(10))
    assert result["source"] == "full_solver"
    assert result["confidence"].level == "low"
    assert result["result"]["case"] == 10
    assert "fields" in result["result"]


def test_solver_lookup_matches_the_dns_ground_truth_exactly():
    # solver_lookup IS the DNS ground truth (not an approximation of it),
    # so this should be an exact match, not just "close".
    sc = SliceCase.load(10, "xy_slice_1", None, DESCRIPTORS)
    looked_up = solver_lookup(_case_params(10))
    for i, field in enumerate(("u", "T", "cp")):
        np.testing.assert_array_equal(looked_up["fields"][field], sc.conv_rms[i])


def test_solver_lookup_rejects_a_novel_operating_point():
    novel = {"Pb_Pc": 3.3, "Thw_Tc": 1.25, "Tcw_Tc": 0.87}
    with pytest.raises(KeyError, match="No precomputed DNS case matches"):
        solver_lookup(novel)


def test_confidence_score_matches_route_decision():
    for case in (1, 10, 15):
        conf = confidence_score(_case_params(case))
        result = route(_case_params(case))
        expected_source = "surrogate" if conf.trusts_surrogate() else "full_solver"
        assert result["source"] == expected_source

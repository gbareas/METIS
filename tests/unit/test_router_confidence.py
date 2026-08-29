import pytest

from metis.router.confidence import confidence_score

# case01-09 training grid + case10/case15 OOD, from
# pub4_grassmann_rom/results/case_descriptors.json (verified against the
# live file when this module was written).
TRAINING_CASES = {
    "case01": {"Pb_Pc": 1.5, "Thw_Tc": 1.1, "Tcw_Tc": 0.95},
    "case02": {"Pb_Pc": 1.5, "Thw_Tc": 1.2, "Tcw_Tc": 0.90},
    "case03": {"Pb_Pc": 1.5, "Thw_Tc": 1.4, "Tcw_Tc": 0.80},
    "case04": {"Pb_Pc": 2.0, "Thw_Tc": 1.1, "Tcw_Tc": 0.95},
    "case05": {"Pb_Pc": 2.0, "Thw_Tc": 1.2, "Tcw_Tc": 0.90},
    "case06": {"Pb_Pc": 2.0, "Thw_Tc": 1.4, "Tcw_Tc": 0.80},
    "case07": {"Pb_Pc": 5.0, "Thw_Tc": 1.1, "Tcw_Tc": 0.95},
    "case08": {"Pb_Pc": 5.0, "Thw_Tc": 1.2, "Tcw_Tc": 0.90},
    "case09": {"Pb_Pc": 5.0, "Thw_Tc": 1.4, "Tcw_Tc": 0.80},
}
CASE10 = {"Pb_Pc": 1.5, "Thw_Tc": 1.185, "Tcw_Tc": 1.035}
CASE15 = {"Pb_Pc": 1.5, "Thw_Tc": 1.132, "Tcw_Tc": 0.982}


@pytest.mark.parametrize("case_id,params", TRAINING_CASES.items())
def test_all_training_cases_score_high(case_id, params):
    conf = confidence_score(params)
    assert conf.level == "high", f"{case_id}: expected high, got {conf.level}"
    assert conf.in_envelope
    assert conf.trusts_surrogate()


def test_case10_flagged_low_matches_known_collapse():
    conf = confidence_score(CASE10)
    assert conf.level == "low"
    assert not conf.trusts_surrogate()
    assert "case10" in conf.reason


def test_case15_flagged_medium_matches_known_mild_excursion():
    conf = confidence_score(CASE15)
    assert conf.level == "medium"
    assert not conf.trusts_surrogate()
    assert "case15" in conf.reason


def test_case10_scores_strictly_lower_confidence_than_case15():
    # The whole point of this diagnostic: distinguish these two OOD cases,
    # which both sit inside the (Pb_Pc, Thw_Tc) envelope the surrogate is
    # conditioned on and would look identically "OOD" under a naive
    # distance-in-conditioning-space rule.
    levels = ["low", "medium", "high"]
    assert levels.index(confidence_score(CASE10).level) < levels.index(
        confidence_score(CASE15).level
    )


def test_pb_pc_out_of_range_with_in_range_tcw_is_medium_not_high():
    params = {"Pb_Pc": 10.0, "Thw_Tc": 1.2, "Tcw_Tc": 0.90}
    conf = confidence_score(params)
    assert conf.level == "medium"
    assert not conf.in_envelope
    assert not conf.trusts_surrogate()


def test_missing_key_raises():
    with pytest.raises(ValueError, match="Tcw_Tc"):
        confidence_score({"Pb_Pc": 1.5, "Thw_Tc": 1.1})

"""Regression test for the frozen `regime-v1` benchmark (milestone R7.2).

Runs on a small committed feature artifact
(`tests/data/regime_v1/block_features.npz`, ~60 KB, extracted once from
the real case01-09/10/15 DNS) — pure numpy, no DNS access, CI-safe. If
this fails, either a deterministic pipeline changed or the science moved;
in the latter case update configs/benchmarks/regime_discovery_v1.yaml and
add a FINDINGS.md entry.
"""
from pathlib import Path

import numpy as np
import pytest

from metis.evaluation.benchmark import BenchmarkResult, load_config, run_regime_v1
from metis.features.regime import FEATURE_BLOCK_NAMES, CaseGridLabel

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "regime_v1" / "block_features.npz"


@pytest.fixture(scope="module")
def result() -> BenchmarkResult:
    with np.load(FIXTURE) as npz:
        blocks = {b: npz[b] for b in FEATURE_BLOCK_NAMES}
        labels = {
            str(c): CaseGridLabel(Pb_Pc=float(p), Thw_Tc=float(t))
            for c, p, t in zip(npz["case_ids"], npz["Pb_Pc"], npz["Thw_Tc"])
        }
    return run_regime_v1(blocks, labels, load_config())


def test_all_frozen_checks_pass(result):
    result.assert_passes()


def test_bulk_is_the_pressure_specialist(result):
    assert max(FEATURE_BLOCK_NAMES, key=lambda b: result.blocks[b]["ari_vs_Pb_Pc"]) == "bulk"
    assert max(FEATURE_BLOCK_NAMES, key=lambda b: result.blocks[b]["loco_accuracy_Pb_Pc"]) == "bulk"


def test_rms_profile_is_the_thermal_specialist(result):
    assert max(FEATURE_BLOCK_NAMES, key=lambda b: result.blocks[b]["ari_vs_Thw_Tc"]) == "rms_profile"
    assert max(FEATURE_BLOCK_NAMES, key=lambda b: result.blocks[b]["loco_accuracy_Thw_Tc"]) == "rms_profile"


def test_mfa_restores_ood_pressure_where_naive_fails(result):
    assert result.combined["mfa"]["ood_nearest_Pb_Pc_centroid"] == {"case10": 1.5, "case15": 1.5}
    assert result.combined["naive"]["ood_nearest_Pb_Pc_centroid"] == {"case10": 5.0, "case15": 5.0}


def test_no_metric_drift(result):
    exp = load_config()["expected"]["blocks"]
    for b in FEATURE_BLOCK_NAMES:
        for k, want in exp[b].items():
            assert result.blocks[b][k] == pytest.approx(want, abs=1e-6)

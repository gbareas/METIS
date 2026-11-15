"""Regression test for the standard-physics numbers (Stage 3 / §24).

`tests/data/physics_regression/bulk_reference.json` holds the bulk
dimensionless groups and wall friction Reynolds numbers published in
`pub4_grassmann_rom/results/case_setup_table.json` for the 11 simulated
cases. The committed `tests/data/regime_v1/block_features.npz` `bulk`
block was extracted from the same real DNS by
`metis.features.regime.build_feature_matrix_block(..., "bulk")`, i.e. by
`metis.features.physics.case_physics_summary`.

If this fails, METIS's physics layer no longer reproduces the published
values — investigate before touching the reference.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from metis.features.regime import ALL_CASE_IDS, BULK_FEATURE_NAMES

DATA = Path(__file__).resolve().parents[1] / "data"
REFERENCE = json.loads((DATA / "physics_regression" / "bulk_reference.json").read_text())


@pytest.fixture(scope="module")
def bulk_block():
    with np.load(DATA / "regime_v1" / "block_features.npz") as npz:
        return {str(c): row for c, row in zip(npz["case_ids"], npz["bulk"])}


@pytest.mark.parametrize("case_id", ALL_CASE_IDS)
def test_bulk_quantities_match_published_pub4_values(bulk_block, case_id):
    got = dict(zip(BULK_FEATURE_NAMES, bulk_block[case_id]))
    want = REFERENCE[case_id]
    for name in BULK_FEATURE_NAMES:
        assert got[name] == pytest.approx(want[name], rel=1e-9), name


def test_reference_covers_every_regime_case():
    assert set(REFERENCE) == set(ALL_CASE_IDS)
    assert all(k in REFERENCE["case01"] for k in BULK_FEATURE_NAMES)


def test_every_bulk_quantity_is_finite_and_positive(bulk_block):
    for case_id in ALL_CASE_IDS:
        row = bulk_block[case_id]
        assert np.all(np.isfinite(row)) and np.all(row > 0), case_id

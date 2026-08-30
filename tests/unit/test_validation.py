"""Tests for metis.data.validation (R3).

Every failure path is driven by a deliberately corrupted mock fixture
(metis.testing.corrupt) and asserts the report fails *for the right
reason* (the `check` slug), not merely that it fails.
"""
import json
import shutil

import pytest

from metis.data.registry import CaseRegistry
from metis.data.validation import (
    ValidationError,
    validate_case,
    validate_compatibility,
    validate_dns_case,
    validate_slice_case,
)
from metis.testing import corrupt, mock_dns, mock_slices

SLICE_ID = "s3_center"


def _make_case(root, case_id, *, seed=1, nx=6, ny=8, nz=6, **ratios):
    ratios = {"Pb_Pc": 1.5, "Thw_Tc": 1.1, "Tcw_Tc": 0.95, **ratios}
    raw = root / "raw" / case_id
    h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=nx, ny=ny, nz=nz,
                                 seed=seed, case_id=case_id, **ratios)
    proc = root / "processed" / case_id
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    mock_slices.generate(root / "processed_slices" / case_id / SLICE_ID,
                         nx=8, nz=8, n_snapshots=10, seed=seed,
                         case_id=case_id, slice_id=SLICE_ID, **ratios)
    return h5, proc / "metadata.json"


@pytest.fixture
def root(tmp_path):
    _make_case(tmp_path, "case01")
    return tmp_path


@pytest.fixture
def desc(root):
    return CaseRegistry(root)["case01"]


def _checks(report):
    return {i.check for i in report.issues}


# --- clean data -----------------------------------------------------
def test_clean_case_passes_with_no_errors(desc):
    report = validate_case(desc)
    assert report.ok
    assert not report.errors


def test_clean_slice_passes(desc):
    report = validate_slice_case(desc.load_slice(SLICE_ID))
    assert report.ok, report.summary()


# --- structural failures ------------------------------------------
def test_grid_mismatch_is_reported(root, desc):
    corrupt.break_grid_metadata(desc.metadata_path, Nx=999)
    report = validate_case(CaseRegistry(root)["case01"])
    assert not report.ok
    assert "grid_matches_metadata" in _checks(report)


def test_unsorted_timesteps_flagged_without_loading(root, desc):
    corrupt.unsort_timesteps(desc.metadata_path)
    report = validate_case(CaseRegistry(root)["case01"], load=False)
    assert not report.ok
    assert "timesteps_ordered" in _checks(report)


def test_missing_raw_is_a_warning_not_an_error(root, desc):
    for h5 in (root / "raw" / "case01").glob("*.h5"):
        h5.unlink()
    report = validate_case(desc)
    assert report.ok  # only a warning
    assert "raw_present" in {i.check for i in report.warnings}


# --- numerical failures -----------------------------------------
def test_nan_in_field_is_reported_with_field_name(desc):
    corrupt.inject_nan(desc.raw_snapshot(), field="T", n=2)
    report = validate_dns_case(desc.load())
    assert not report.ok
    errs = [i for i in report.errors if i.check == "no_nan"]
    assert errs and "T" in errs[0].message


def test_nonpositive_density_is_reported(desc):
    corrupt.make_nonpositive(desc.raw_snapshot(), field="rho")
    report = validate_dns_case(desc.load())
    assert "physically_positive" in {i.check for i in report.errors}


def test_slice_nan_is_reported(desc):
    corrupt.inject_slice_nan(desc.slice_dir(SLICE_ID), field="T")
    report = validate_slice_case(desc.load_slice(SLICE_ID))
    assert "no_nan" in {i.check for i in report.errors}


def test_duplicate_slice_frame_is_a_warning(desc):
    corrupt.duplicate_slice_frame(desc.slice_dir(SLICE_ID), field="u", src=0, dst=1)
    report = validate_slice_case(desc.load_slice(SLICE_ID))
    assert report.ok  # warning, not error
    assert "no_duplicate_snapshots" in {i.check for i in report.warnings}


def test_slice_grid_mismatch_reported(desc):
    meta_path = desc.slice_dir(SLICE_ID) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["grid"]["NX"] = 4321
    meta_path.write_text(json.dumps(meta))
    report = validate_slice_case(desc.load_slice(SLICE_ID))
    assert "slice_grid_matches_metadata" in {i.check for i in report.errors}


# --- cross-case compatibility --------------------------------------
def test_compatible_cases_pass(tmp_path):
    _make_case(tmp_path, "case01", seed=1)
    _make_case(tmp_path, "case02", seed=2)
    reg = CaseRegistry(tmp_path)
    assert validate_compatibility([reg["case01"], reg["case02"]]).ok


def test_grid_mismatch_across_cases_reported(tmp_path):
    _make_case(tmp_path, "case01", seed=1, nx=6, ny=8, nz=6)
    _make_case(tmp_path, "case02", seed=2, nx=8, ny=8, nz=6)
    reg = CaseRegistry(tmp_path)
    report = validate_compatibility([reg["case01"], reg["case02"]])
    assert "grids_match" in {i.check for i in report.errors}


def test_field_name_mismatch_across_loaded_cases_reported(tmp_path):
    _make_case(tmp_path, "case01", seed=1)
    _make_case(tmp_path, "case02", seed=2)
    reg = CaseRegistry(tmp_path)
    a, b = reg["case01"].load(), reg["case02"].load()
    b.fields.pop(next(iter(b.fields)))
    report = validate_compatibility([a, b])
    assert "field_names_match" in {i.check for i in report.errors}


# --- report object -----------------------------------------------
def test_report_raise_if_failed_and_bool(desc):
    corrupt.inject_nan(desc.raw_snapshot(), field="T")
    report = validate_dns_case(desc.load())
    assert not bool(report)
    assert "case01" in report.summary()
    with pytest.raises(ValidationError):
        report.raise_if_failed()

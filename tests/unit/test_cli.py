"""Tests for the `metis` CLI (I7).

Only subcommands that need nothing beyond raw HDF5 (mock_dns) are
exercised end to end, so this runs in CI without torch / DNS data.
"""
import json
import shutil

import pytest

from metis.cli import main
from metis.testing import corrupt, mock_dns


def _make_case(root, case_id, seed):
    raw = root / "raw" / case_id
    _h5, meta = mock_dns.generate(raw / f"s_{seed}.h5", nx=6, ny=8, nz=6, seed=seed,
                                  case_id=case_id, Pb_Pc=1.5 + seed, Thw_Tc=1.1, Tcw_Tc=0.95)
    proc = root / "processed" / case_id
    proc.mkdir(parents=True)
    shutil.copy(meta, proc / "metadata.json")
    meta.unlink()
    return proc / "metadata.json"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    _make_case(tmp_path, "case01", 1)
    _make_case(tmp_path, "case02", 2)
    monkeypatch.setenv("METIS_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_version_and_help_exit_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "metis" in capsys.readouterr().out


def test_no_command_prints_help_and_returns_1(capsys):
    assert main([]) == 1
    assert "usage: metis" in capsys.readouterr().out


def test_cases_list(data_root, capsys):
    assert main(["cases", "list"]) == 0
    out = capsys.readouterr().out
    assert "case01" in out and "case02" in out and "Pb_Pc" in out


def test_case_validate_ok_and_unknown(data_root, capsys):
    assert main(["case", "validate", "case01"]) == 0
    assert "case01: OK" in capsys.readouterr().out
    assert main(["case", "validate", "nope"]) == 2
    assert "unknown case 'nope'" in capsys.readouterr().err


def test_case_validate_fails_on_corrupted_metadata(data_root):
    corrupt.break_grid_metadata(data_root / "processed" / "case01" / "metadata.json", Nx=999)
    assert main(["case", "validate", "case01"]) == 1


def test_analyze_physics(data_root, capsys):
    assert main(["analyze", "physics", "case01"]) == 0
    assert "physics(case01)" in capsys.readouterr().out


def test_dataset_build_writes_artifact(data_root, tmp_path, capsys):
    out = tmp_path / "ds"
    assert main(["dataset", "build", "--feature-set", "bulk",
                 "--cases", "case01,case02", "--out", str(out)]) == 0
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["feature_set"] == "bulk" and meta["provenance"]["n_cases"] == 2


def test_report_regime_v1_from_json(tmp_path, capsys):
    sample = {
        "name": "regime-v1", "passed": True,
        "checks": [{"name": "c", "passed": True, "detail": "ok"}],
        "blocks": {"bulk": {"ari_vs_Pb_Pc": 0.3, "ari_vs_Thw_Tc": 0.1,
                            "loco_accuracy_Pb_Pc": 0.5, "loco_accuracy_Thw_Tc": 0.2}},
        "combined": {"mfa": {"ood_nearest_Pb_Pc_centroid": {"case10": 1.5}},
                     "naive": {"ood_nearest_Pb_Pc_centroid": {"case10": 5.0}}},
    }
    src = tmp_path / "regime_v1.json"
    src.write_text(json.dumps(sample))
    out = tmp_path / "rep"
    assert main(["report", "regime-v1", "--from", str(src), "--out", str(out)]) == 0
    assert (out / "summary.md").exists() and (out / "metrics.json").exists()


def test_missing_data_root_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("METIS_DATA_ROOT", raising=False)
    assert main(["cases", "list", "--config", "/nonexistent/x.yaml"]) in (2,)

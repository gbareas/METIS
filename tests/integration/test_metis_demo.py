"""`metis demo` must run green on a clean checkout (it's what the Docker
image and the README's quick start run)."""
import json

import pytest

from metis.cli import main


def test_demo_runs_and_passes(tmp_path):
    pytest.importorskip("matplotlib")
    rc = main(["demo", "--out", str(tmp_path)])
    assert rc == 0
    bench = json.loads((tmp_path / "regime_v1.json").read_text())
    assert bench["passed"]
    routes = json.loads((tmp_path / "router_decisions.json").read_text())
    by_name = {r["scenario"]: r for r in routes}
    assert by_name["case10 (OOD)"]["routed_to"] == "DNS database"
    assert by_name["case05"]["routed_to"] == "surrogate"
    assert (tmp_path / "reports" / "regime-v1" / "summary.md").exists()
    assert (tmp_path / "summary.png").exists()

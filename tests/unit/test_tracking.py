"""Tests for metis.tracking (I1)."""

import pytest

from metis import tracking

mlflow = pytest.importorskip("mlflow")


def test_flatten_params_dots_nested_keys():
    flat = tracking.flatten_params({"a": 1, "b": {"c": 2, "d": {"e": 3}}, "l": [1, 2]})
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3, "l": [1, 2]}


def test_disabled_run_is_a_noop_null_handle():
    with tracking.run("x", params={"a": 1}, enabled=False) as r:
        assert r.active is False and r.run_id is None
        r.log_metrics({"m": 1.0})          # must not raise
        r.log_artifact("/nonexistent")
        r.log_dict({"k": 1}, "d.json")


@pytest.fixture
def mlruns(tmp_path):
    return tmp_path / "mlruns"


def test_run_logs_params_metrics_tags_and_artifacts(tmp_path, mlruns, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    art = tmp_path / "note.txt"
    art.write_text("hello")

    with tracking.run(
        "test-exp", run_name="r0",
        params={"model": {"kind": "ae", "latent": 4}, "seed": 0},
        tags={"kind": "unit"},
        tracking_dir=mlruns,
    ) as r:
        assert r.active and r.run_id
        r.log_metrics({"loss": 0.5, "val_loss": 0.7})
        r.log_dict({"ok": True}, "result.json")
        r.log_artifact(art, artifact_path="notes")
        run_id = r.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=mlruns.as_uri())
    run = client.get_run(run_id)

    assert run.data.params["model.kind"] == "ae"
    assert run.data.params["model.latent"] == "4"
    assert run.data.params["seed"] == "0"
    assert run.data.metrics["loss"] == 0.5
    assert "runtime_seconds" in run.data.metrics          # auto-logged on exit
    assert run.data.tags["git.commit"]                    # auto-logged
    assert run.data.tags["metis.version"]
    assert run.data.tags["kind"] == "unit"

    artifacts = {f.path for f in client.list_artifacts(run_id)}
    assert "result.json" in artifacts
    assert "notes" in artifacts


def test_acceptance_run_id_recovers_data_config_metrics(tmp_path, mlruns, monkeypatch):
    """§13 acceptance: from a run id, another user can see the data used,
    the config, the metrics, and where artifacts are."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with tracking.run(
        "acc", params={"data_root": "/data/dns", "benchmark": {"name": "regime-v1"}},
        tracking_dir=mlruns,
    ) as r:
        r.log_metrics({"ari_vs_Pb_Pc": 0.357})
        r.log_dict({"blocks": {"bulk": {"ari_vs_Pb_Pc": 0.357}}}, "regime_v1.json")
        run_id = r.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=mlruns.as_uri())
    run = client.get_run(run_id)
    assert run.data.params["data_root"] == "/data/dns"
    assert run.data.params["benchmark.name"] == "regime-v1"
    assert run.data.metrics["ari_vs_Pb_Pc"] == pytest.approx(0.357)
    assert "regime_v1.json" in {f.path for f in client.list_artifacts(run_id)}


def test_exception_in_run_still_records_the_run(tmp_path, mlruns, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError), tracking.run("boom", tracking_dir=mlruns) as r:
        rid = r.run_id
        raise RuntimeError("kaboom")

    run = mlflow.tracking.MlflowClient(tracking_uri=mlruns.as_uri()).get_run(rid)
    assert run.info.status == "FAILED"
    assert "runtime_seconds" in run.data.metrics

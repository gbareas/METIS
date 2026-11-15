"""Experiment tracking (milestone I1).

A thin wrapper over MLflow so every ML / discovery run is reproducible:
given a run id, another user can see what data were used, how the run was
configured, what metrics came out, and where the artifacts are.

Design:
  - local-first: runs go to `<repo>/mlruns/` unless `$MLFLOW_TRACKING_URI`
    is set; no server or database.
  - optional: `mlflow` lives in the `ml` extra. If it isn't installed,
    `run(...)` yields a no-op handle so callers never have to branch.
  - opinionated defaults: git commit, metis version, python/platform, and
    the flattened config are logged automatically; runtime is logged on
    exit.

Usage::

    from metis import tracking
    with tracking.run("regime-v1", params=config) as r:
        r.log_metrics({"ari_vs_Pb_Pc": 0.357})
        r.log_dict(results, "results.json")
"""
from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from metis import __version__
from metis.config import REPO_ROOT, git_commit

# Local-first: a plain mlruns/ directory, no server or database (I1).
# MLflow >= 3 gates the file store behind this opt-out ("maintenance
# mode"); acceptable for v1. `mlflow ui` needs the same flag exported.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:  # optional dependency
    import mlflow as _mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the ml extra
    _mlflow = None
    _MLFLOW_AVAILABLE = False

DEFAULT_TRACKING_DIR = REPO_ROOT / "mlruns"
_MLFLOW_PARAM_MAXLEN = 500


def is_available() -> bool:
    """True if MLflow is installed (the `ml` extra)."""
    return _MLFLOW_AVAILABLE


def flatten_params(obj: Mapping, prefix: str = "") -> dict[str, object]:
    """Flatten a nested config to dotted keys, e.g. {'a': {'b': 1}} ->
    {'a.b': 1}. Lists become their repr (MLflow params are flat scalars)."""
    flat: dict[str, object] = {}
    for key, value in obj.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


class _Run:
    """Handle for an active MLflow run."""

    active = True

    def __init__(self, run):
        self._run = run
        self.run_id: str = run.info.run_id

    def log_params(self, params: Mapping) -> None:
        flat = {
            k: (str(v)[:_MLFLOW_PARAM_MAXLEN] if not isinstance(v, (int, float, bool)) else v)
            for k, v in flatten_params(params).items()
        }
        _mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        _mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        _mlflow.log_metric(key, float(value), step=step)

    def set_tags(self, tags: Mapping[str, object]) -> None:
        _mlflow.set_tags(dict(tags))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        _mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, obj, name: str) -> None:
        _mlflow.log_dict(obj, name)

    def log_text(self, text: str, name: str) -> None:
        _mlflow.log_text(text, name)


class _NullRun:
    """No-op stand-in when MLflow isn't installed. Same surface as `_Run`."""

    active = False
    run_id = None

    def log_params(self, *a, **k) -> None: ...
    def log_metrics(self, *a, **k) -> None: ...
    def log_metric(self, *a, **k) -> None: ...
    def set_tags(self, *a, **k) -> None: ...
    def log_artifact(self, *a, **k) -> None: ...
    def log_dict(self, *a, **k) -> None: ...
    def log_text(self, *a, **k) -> None: ...


@contextmanager
def run(
    experiment: str,
    *,
    run_name: str | None = None,
    params: Mapping | None = None,
    tags: Mapping | None = None,
    tracking_dir: str | Path | None = None,
    enabled: bool = True,
) -> Iterator[_Run | _NullRun]:
    """Context manager for one tracked run.

    Auto-logs: git commit, metis version, python/platform, the flattened
    `params`, and (on exit) `runtime_seconds`. `enabled=False` or a
    missing MLflow install yields a `_NullRun` that silently no-ops.
    """
    if not enabled or not _MLFLOW_AVAILABLE:
        yield _NullRun()
        return

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        _mlflow.set_tracking_uri(
            (Path(tracking_dir) if tracking_dir else DEFAULT_TRACKING_DIR).as_uri()
        )
    _mlflow.set_experiment(experiment)

    started = time.time()
    with _mlflow.start_run(run_name=run_name) as active:
        handle = _Run(active)
        handle.set_tags({
            "metis.version": __version__,
            "git.commit": git_commit(),
            "python.version": platform.python_version(),
            "platform": platform.platform(),
            "python.executable": sys.executable,
            **dict(tags or {}),
        })
        if params:
            handle.log_params(params)
        try:
            yield handle
        finally:
            handle.log_metric("runtime_seconds", time.time() - started)

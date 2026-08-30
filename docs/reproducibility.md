# Reproducibility

Every layer that produces an artifact records enough to reproduce it.

## Configuration, not code edits

No environment-specific path lives in `src/`, `scripts/`, or
`pyproject.toml`. The DNS data root resolves as **`--data-root` > YAML
`data.root` (`--config`, else `configs/default.yaml`) > `$METIS_DATA_ROOT`
> a clear error**. The regime scripts, the CLI, and every runner use the
same resolver (`metis.config`).

## Feature datasets are fingerprinted and cached

`build_feature_dataset(registry, case_ids, feature_set, out_dir=)` writes
`<out_dir>/{data.npz, metadata.json}`. `metadata.json.provenance`
carries a **`fingerprint`** = sha256 of `(case_ids, feature_set, source
of features/{regime,physics,pod}.py)`, plus `git commit`, `data_root`,
timestamp, and shape. A re-run with a matching fingerprint **loads the
artifact instead of re-reading raw DNS**. Editing feature code changes
the fingerprint; editing the underlying DNS files does **not** — pass
`--rebuild` (or delete the artifact) in that case.

## Experiment tracking (MLflow)

`metis.tracking.run(experiment, params=, tags=)` is a context manager
over MLflow. It writes to `<repo>/mlruns/` (no server, no database)
unless `$MLFLOW_TRACKING_URI` is set, and auto-logs: git commit,
`metis.__version__`, python/platform, the flattened `params`, and
`runtime_seconds` on exit. Without the `ml` extra it yields a no-op
handle so callers never branch.

Given a run id, another user can recover the data used, the config, the
metrics, and the artifact locations:

```bash
metis report regime-v1 --run-id <run_id>     # pulls the result JSON from the run
```

`mlflow ui` needs `MLFLOW_ALLOW_FILE_STORE=true` exported (MLflow ≥ 3
gates the file store).

## The frozen benchmark

`configs/benchmarks/regime_discovery_v1.yaml` pins the settled regime
result (FINDINGS.md §1–4): which block is the pressure vs. thermal
specialist, the MFA sanity check, and every metric value at 1e-6
tolerance. `metis benchmark regime-v1` reproduces it against real data
and **exits non-zero on any regression**;
`tests/regression/test_regime_v1_benchmark.py` runs the same checks on a
60 KB committed feature artifact, so CI needs no DNS. **Changing an
expected value means the science changed — add a `FINDINGS.md` entry.**

## Provenance on every artifact

- `AnalysisResult` / `FeatureDataset` / benchmark JSON: `created_at`,
  `code_version` (git commit), `data_root`, shape/fingerprint.
- `Report`: `generated_at`, `metis_version`, `git_commit` in
  `summary.md` and `metrics.json`.
- `ArtifactRegistry` entries: `created_at` / `updated_at`, `git_commit`,
  `metis_version`, `run_id`, `dataset_id`, `status`, `scope`.

## Determinism

- `kmeans_cluster` uses a fixed seed; the regime pipeline is
  deterministic given the same feature matrix.
- `Trainer` re-seeds torch + numpy before the loop; a fixed `seed`
  reproduces the same weights and `best_train_mse`.
- Mock generators (`metis.testing.mock_dns` / `mock_slices`) are seeded.

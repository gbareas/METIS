# Architecture

METIS is a pipeline of thin layers between a RHEA DNS snapshot and a
report. Each layer has one job and a small, stable interface; the CLI and
scripts are shims over library calls.

```
RHEA DNS output  (raw HDF5 + companion metadata.json + slice products)
        │
        ▼
  data.ingestion     HDF5Reader → DNSCase ,  SliceReader → SliceCase
        │
        ▼
  data.registry      CaseRegistry / CaseDescriptor — one authoritative
        │            representation of a case; resolves paths, loads data
        ▼
  data.validation    validate_case / validate_dns_case / validate_slice_case
        │            / validate_compatibility → ValidationReport (ok/errors/warnings)
        ▼
  data.preprocessing StandardScaler / MinMaxScaler (fit-on-train),
        │            strip_ghost_cells / select_variables / subsample
        ▼
  features           physics (bulk groups, wall Re_tau, profiles),
        │            spectra (wavenumber PSD), pod (snapshot POD),
        │            regime (feature blocks + labels)
        ▼
  data.datasets      build_feature_dataset → FeatureDataset (Level-1,
        │            one row per case) + fingerprinted artifact cache
        ▼
  analysis           run_analysis(registry, kind, case) → AnalysisResult
        │            {physics, spectra, pod, regime_features}
        ▼
  models + training  RepresentationModel (PCARepresentation, Autoencoder);
        │            Trainer (Adam + early stop + checkpoint + MLflow)
        ▼
  evaluation         metrics (MAE/RMSE/nRMSE/relL2/R²),
        │            physical (profile/spectrum/POD agreement),
        │            representation (ARI/LOCO, latent stability & correlation),
        │            assessment.assess_model (the §17 "is it better?" gate),
        │            regime + benchmark (frozen regime-v1)
        ▼
  reporting          Report → reports/<name>/{summary.md, metrics.json, figures/}
```

Cross-cutting:

- **`config`** — data-root resolution (`--data-root` > YAML > env),
  `git_commit()` for provenance.
- **`tracking`** — `run(experiment, params=, ...)` context manager over
  MLflow (local `mlruns/`, no server); a no-op handle when `ml` isn't
  installed.
- **`registry`** — `ArtifactRegistry` over `artifacts/registry.json`:
  which datasets/models exist and whether they're `validated`.
- **`cli`** — `metis <command>`; every subcommand wraps one of the above.
- **`testing`** — `mock_dns` / `mock_slices` generate DNS-shaped
  fixtures; `corrupt` mutates them for the validation tests.

## Module map (`src/metis/`)

```
config.py            data-root / config / git-commit resolution
tracking.py          MLflow wrapper (optional)
cli.py               the `metis` command
analysis.py          run_analysis + AnalysisResult

data/
  ingestion/         hdf5_reader.py (DNSCase), slice_reader.py (SliceCase)
  registry.py        CaseRegistry, CaseDescriptor
  validation/        report.py, checks.py
  preprocessing/     base.py (Transform), scalers.py, fields.py
  datasets/          feature_dataset.py, build.py (fingerprinted cache)

features/            physics.py, spectra.py, pod.py, regime.py
models/              base.py (RepresentationModel), pca.py, autoencoder.py
training/            trainer.py
evaluation/          metrics.py, physical.py, representation.py,
                     assessment.py, regime.py, benchmark.py (regime-v1)
reporting/           report.py, generators.py
registry/            artifact_registry.py

router/              Track C/D — confidence.py, surrogate.py, solver.py,
                     core.py, agent.py, demo.py (see docs/router_agent.md)
testing/             mock_dns.py, mock_slices.py, corrupt.py

api/ inference/ monitoring/   reserved namespaces (roadmap v1 §17/§20)
```

## Design rules

- **Validated numerical code is not rewritten** to look more abstract —
  interfaces wrap it, regression tests pin it.
- **Configuration over script editing** — no environment-specific path in
  `src/`, `scripts/`, or `pyproject.toml`.
- **Analysis code asks the registry for a case** — it never reconstructs
  paths.
- **Fit preprocessing on training cases, apply to held-out ones.**
- **Every result records** source cases, config, git commit, seed, and
  where its artifacts are.
- **A model is only "better" if the physical diagnostics agree**, not
  just one ML metric (`evaluation.assessment`).

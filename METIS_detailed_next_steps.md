# METIS — Detailed Refactoring & Implementation Plan

**Status baseline:** repository snapshot inspected 2026-08-30  
**Project:** METIS — Machine-learning Engine for Transcritical Insight & Statistics  
**Core identity:** RHEA generates the physics; METIS extracts the insight.

---

# 1. Purpose of this document

This document converts the current METIS repository into a concrete implementation backlog.

It is intentionally narrower and more operational than the long-term roadmap. It answers four questions:

1. **What is already implemented and should be preserved?**
2. **What needs refactoring before the code becomes genuinely reusable by the group?**
3. **What new capabilities should be implemented next?**
4. **What should explicitly be deferred to avoid scope creep?**

The target is a framework that future BSc/MSc/PhD students can use to ingest RHEA DNS output, reproduce trusted physical analyses, run data-driven discovery methods, compare cases, and extend the platform without rebuilding the infrastructure.

The parallel router-agent portfolio module remains in the repository, but it should not define the core architecture of METIS.

---

# 2. Current repository baseline

## 2.1 Already implemented and validated

The following components are real implementations and should be treated as the current stable core.

### DNS ingestion

- `metis.data.ingestion.hdf5_reader`
  - reads the real RHEA flat-HDF5 snapshot layout;
  - handles ghost cells;
  - uses companion processed metadata;
  - has been verified against real DNS data.

- `metis.data.ingestion.slice_reader`
  - reads the time-resolved 2D homogeneous-plane slice product;
  - exposes data through `SliceCase`;
  - has been exercised on the existing processed slice database.

### Standard physical analysis

- `metis.features.physics`
  - wall-normal profiles;
  - bulk quantities;
  - friction Reynolds numbers;
  - case-level physics summary;
  - cross-checked against existing published calculations.

- `metis.features.spectra`
  - 1D spatial wavenumber spectra;
  - premultiplied spectra;
  - validated with Parseval consistency on real data.

- `metis.features.pod`
  - method-of-snapshots POD;
  - modal energy fractions;
  - mode-field reconstruction;
  - validated against prior Pub 4 results.

### Regime-discovery reference workflow

- `metis.features.regime`
  - compact and rich feature sets;
  - block-level feature construction;
  - case operating-condition labels.

- `metis.evaluation.regime`
  - standardization;
  - PCA;
  - deterministic k-means;
  - adjusted Rand index;
  - leave-one-case-out nearest-centroid evaluation;
  - MFA-style block weighting.

- scripts:
  - `run_regime_discovery.py`
  - `run_regime_ablation.py`
  - `run_regime_blockwise.py`

- recorded outputs:
  - `results/regime_discovery.json`
  - `results/regime_discovery_ablation.json`
  - `results/regime_discovery_blockwise.json`

### Research methodology

- `research_protocol.md`
- `FINDINGS.md`

The regime-discovery work has already reached an important methodological conclusion:

- pressure information is primarily carried by the `bulk` block;
- thermal information is primarily carried by the `rms_profile` block;
- `mean_profile` is not useful for the current reference task;
- POD energy fractions do not improve the current regime-discovery objective;
- one shared `k=3` clustering cannot simultaneously reproduce two independent 3-level factorial axes.

This result should be considered **settled evidence**, not an invitation to keep tuning the same clustering indefinitely.

### Testing / CI

Existing repository structure includes:

- unit tests;
- integration tests;
- synthetic/mock DNS generators;
- GitHub Actions CI;
- Ruff;
- pytest.

The self-contained core test suite passes when the package is correctly importable.

### Router-agent portfolio module

`metis.router` already contains:

- deterministic OOD/confidence logic;
- surrogate wrapper;
- precomputed DNS fallback;
- routing core;
- Claude tool wrapper;
- curated scenarios;
- Streamlit demo.

This is useful portfolio work, but it is architecturally separate from the group infrastructure.

---

# 3. Current architectural weaknesses

The current repository is scientifically stronger than its software architecture suggests. The main gaps are not missing physics methods; they are **portability, configuration, dataset abstraction, experiment lifecycle, and extensibility**.

## 3.1 Hard-coded local paths

Current regime scripts contain absolute paths such as:

```python
DATA_ROOT = Path("/home/brinkman/Documents/PostDoc_phase/data")
```

The router also contains an absolute dependency path to the external neural-operator repository.

### Why this matters

A reusable group framework cannot require source-code edits when moved to:

- another workstation;
- another user's account;
- an HPC login node;
- a student's laptop;
- CI;
- a Docker container.

### Required refactor

All environment-specific paths must move out of Python source.

Preferred hierarchy:

1. explicit CLI argument;
2. YAML configuration;
3. environment variable;
4. documented project default where safe.

No scientific module should know the author's home-directory layout.

---

# 4. Target architecture

The intended architecture is:

```text
RHEA / DNS output
       │
       ▼
┌──────────────────┐
│ Case registration │
│ metadata + paths  │
└─────────┬────────┘
          ▼
┌──────────────────┐
│ Validation / QA   │
└─────────┬────────┘
          ▼
┌──────────────────┐
│ Preprocessing     │
│ standardized data │
└─────────┬────────┘
          │
          ├───────────────┐
          ▼               ▼
┌──────────────────┐  ┌────────────────────┐
│ Standard physics │  │ ML dataset builders│
│ stats/POD/spectra│  │ case/local/temporal│
└─────────┬────────┘  └──────────┬─────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌────────────────────┐
          │ Experiments / ML   │
          │ discovery modules  │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Evaluation         │
          │ physics + ML + OOD │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Artifacts / reports│
          │ reproducible runs  │
          └────────────────────┘
```

The first objective is not deployment. The first objective is:

> **A new user can point METIS at an existing DNS case, run validated analyses through configuration, and reproduce a documented result without editing source code.**

---

# 5. Refactoring principles

These should guide every next change.

## 5.1 Preserve validated scientific functions

Do not rewrite working numerical code merely to make it look more abstract.

Refactor interfaces around validated routines while keeping numerical regression tests.

## 5.2 Configuration over script editing

A user should change:

```yaml
cases:
  - case01
  - case02
analysis:
  type: regime_discovery
```

not:

```python
DATA_ROOT = ...
CASE_IDS = ...
```

inside a Python script.

## 5.3 Separate domain objects from filesystem layout

Analysis functions should consume objects such as:

- `DNSCase`
- `SliceCase`
- `CaseDescriptor`
- `Dataset`

rather than repeatedly reconstructing paths.

## 5.4 Separate extraction from evaluation

Example:

```text
DNS → feature extraction → stored feature artifact
                         ↓
                   evaluation
```

Do not force the full HDF5 database to be reread every time a clustering metric changes.

## 5.5 Make every research result traceable

Every result should record enough information to reproduce it:

- source cases;
- metadata/version;
- configuration;
- code commit;
- feature version;
- model parameters;
- random seed;
- metrics;
- output artifact locations.

## 5.6 Notebooks are consumers, not infrastructure

Core logic belongs in `src/metis/`.

Notebooks may visualize or explore outputs, but they should not contain the only implementation of a method.

---

# 6. Milestone R1 — Configuration and path refactor

**Priority:** immediate  
**Type:** refactor  
**Estimated scope:** small/medium  
**Scientific risk:** very low  
**Platform value:** very high

## Goal

Remove machine-specific assumptions and make the existing regime-discovery workflow runnable from another environment.

## Tasks

### R1.1 Add a project configuration layer

Create:

```text
configs/
    default.yaml
    cases/
    analyses/
```

Example:

```yaml
data:
  root: /path/to/data

paths:
  raw: raw
  processed: processed
  slices: processed_slices
  results: results

runtime:
  seed: 0
```

### R1.2 Support environment override

For example:

```bash
export METIS_DATA_ROOT=/path/to/group/data
```

Resolution order:

```text
CLI > config > METIS_DATA_ROOT > error with useful message
```

### R1.3 Remove `DATA_ROOT` constants from scripts

Refactor:

- `run_regime_discovery.py`
- `run_regime_ablation.py`
- `run_regime_blockwise.py`

into CLI entry points.

Example:

```bash
metis regime discover --config configs/analyses/regime.yaml
```

or initially:

```bash
python scripts/run_regime_discovery.py \
    --data-root /path/to/data \
    --output results/regime_discovery.json
```

### R1.4 Remove absolute paths from router code

`src/metis/router/surrogate.py` must not contain a fixed:

```text
/home/.../pub5_neural_operators
```

Possible solutions, in preferred order:

1. package the required surrogate inference interface cleanly;
2. make the external repository an explicitly installed dependency;
3. configure checkpoint/data paths;
4. use an adapter interface around the external project.

### R1.5 Remove local `file:///...` dependency from publishable package metadata

The current router extra is useful locally but not portable.

For a public/general package:

- keep external integration optional;
- document local editable installation separately;
- do not encode one developer's filesystem into `pyproject.toml`.

## Acceptance criteria

- clone repository into a new directory;
- `pip install -e ".[dev]"`;
- point configuration at data;
- run all Track A/B scripts without modifying Python source;
- core tests pass;
- no `/home/<user>/...` paths exist under `src/`, `scripts/`, or `pyproject.toml`.

## Career value

Demonstrates basic software portability and configuration management.

---

# 7. Milestone R2 — Case registry and metadata abstraction

**Priority:** immediate  
**Type:** refactor + foundation

## Goal

Create one authoritative representation of a DNS case.

Currently, metadata knowledge is spread across readers, feature functions, scripts, and external JSON products.

## Proposed objects

### `CaseDescriptor`

Example fields:

```python
@dataclass(frozen=True)
class CaseDescriptor:
    case_id: str
    Pb_Pc: float
    Tcw_Tc: float
    Thw_Tc: float
    nx: int
    ny: int
    nz: int
    n_snapshots: int
    raw_path: Path
    processed_path: Path
    slice_path: Path | None
```

### `CaseRegistry`

Responsibilities:

- discover registered cases;
- load metadata;
- validate required files;
- provide case lookup;
- expose available data products.

Example:

```python
registry = CaseRegistry.from_config(config)

case = registry["case01"]

print(case.Pb_Pc)
print(case.has_slices)
```

## Why this matters

Future analysis code should ask:

> “Give me case01.”

not:

> “Construct `/data/processed/case01/metadata.json`, parse it here, then construct a separate slices path elsewhere.”

## Acceptance criteria

- `case_grid_labels()` can consume descriptors rather than manually reading metadata;
- ingestion readers can be created from a `CaseDescriptor`;
- missing data products generate explicit, useful errors;
- case metadata has one authoritative schema.

---

# 8. Milestone R3 — Data validation layer

**Priority:** immediate  
**Type:** new implementation  
**Target package:** `metis.data.validation`

This directory currently exists but is effectively empty.

## Goal

Catch bad or incompatible DNS data **before** scientific algorithms operate on it.

## Implement

### R3.1 Structural validation

Check:

- required HDF5 fields exist;
- expected array dimensions;
- grid dimensions agree with metadata;
- ghost-cell assumptions are valid;
- snapshots are ordered;
- slice dimensions agree across time;
- required metadata keys exist.

### R3.2 Numerical validation

Check for:

- NaNs;
- infinities;
- impossible dimensions;
- zero/negative values where physically invalid;
- inconsistent coordinates;
- duplicate snapshots;
- unexpected missing timesteps/iterations.

### R3.3 Dataset compatibility validation

Before combining cases:

- same variable definitions;
- compatible grids where required;
- compatible slice locations;
- compatible coordinate normalization;
- same feature layout.

### R3.4 Validation report

Return structured diagnostics rather than only booleans.

Example:

```python
report = validate_case(case)

report.ok
report.errors
report.warnings
```

## Acceptance criteria

The mock data system should include deliberately corrupted fixtures, and tests should verify that validation fails for the correct reason.

---

# 9. Milestone R4 — Preprocessing layer

**Priority:** immediate  
**Type:** new implementation  
**Target package:** `metis.data.preprocessing`

This package is also currently empty.

## Goal

Move common transformations out of individual feature/model implementations.

## Candidate operations

- ghost-cell removal where not already handled at ingestion;
- coordinate normalization;
- variable selection;
- spatial subsampling;
- temporal/window selection;
- scaling/normalization;
- dimensional conversion where required;
- mask generation;
- sensor/probe extraction;
- train-only scaler fitting.

## Critical design rule

Any data-dependent preprocessing used for ML must distinguish:

```text
fit preprocessing on training cases
apply preprocessing to validation/OOD cases
```

Do not standardize using information from held-out cases when testing generalization.

This is particularly important because the current regime evaluation's global feature standardization is acceptable for exploratory small-N structure analysis, but future predictive ML experiments need stricter leakage control.

## Suggested interfaces

```python
transform = StandardScaler()
transform.fit(train_data)
x_train = transform.transform(train_data)
x_test = transform.transform(test_data)
```

or project-specific immutable preprocessing configs.

## Acceptance criteria

- preprocessing operations are deterministic;
- fitted transforms can be serialized;
- inverse transformation exists where meaningful;
- tests verify no held-out leakage.

---

# 10. Milestone R5 — Dataset abstraction and artifact caching

**Priority:** high  
**Type:** new implementation  
**Target package:** `metis.data.datasets`

## Goal

Stop forcing every experiment to regenerate features directly from raw DNS.

Introduce explicit intermediate datasets/artifacts.

## Dataset levels

### Level 1 — Case-level physical features

One row per case.

Examples:

- bulk quantities;
- RMS-derived metrics;
- selected spectral metrics;
- physical metadata.

Useful for:

- regime discovery;
- case similarity;
- sensitivity analysis.

### Level 2 — Local/spatial samples

Samples derived from spatial locations or subdomains.

Useful for:

- local regime classification;
- anomaly detection;
- spatial representation learning.

### Level 3 — Temporal/sensor datasets

Windowed virtual-sensor signals.

Useful for:

- forecasting;
- temporal modelling;
- anomaly detection.

### Level 4 — Full-field datasets

High-dimensional snapshots/sequences.

Useful for:

- autoencoders;
- neural operators;
- field reconstruction/prediction.

## Artifact format

Start simple:

- `.npz`
- `.h5`
- JSON metadata.

Do not add a complex feature store yet.

Each dataset artifact should contain or reference:

- dataset ID;
- source cases;
- features/variables;
- preprocessing;
- split definition;
- generation config;
- code version.

## Example

```bash
metis dataset build \
    --config configs/datasets/regime_v1.yaml
```

Produces:

```text
artifacts/datasets/regime_v1/
    data.npz
    metadata.json
```

## Acceptance criteria

Running an evaluation twice should not require rereading multi-TB raw data if the extracted dataset has not changed.

---

# 11. Milestone R6 — Standard analysis API

**Priority:** high  
**Type:** refactor

## Goal

Turn existing trusted physics functions into a consistent user-facing analysis layer.

## Proposed concept

```python
result = run_analysis(
    case="case01",
    analysis="pod",
    config=...
)
```

or CLI:

```bash
metis analyze physics case01
metis analyze spectra case01 --slice s3_center --field u
metis analyze pod case01 --slice s2_max_u --field u
```

## Standard result object

Each analysis result should expose:

- name;
- input case(s);
- configuration;
- numerical outputs;
- validation metadata;
- artifact-writing method.

## Important constraint

Do **not** force all analyses into one overly generic class hierarchy.

A thin common execution/result protocol is enough.

## Acceptance criteria

At least these existing analyses should run through the same top-level interface:

- physics summary;
- spectra;
- POD;
- regime feature extraction.

---

# 12. Milestone R7 — Freeze regime discovery as a reference benchmark

**Priority:** immediate  
**Type:** consolidation, not new research

## Goal

Turn the completed regime-discovery work into a reproducibility benchmark for METIS.

## Do not

- keep tuning k-means;
- keep adding feature blocks without a new hypothesis;
- force one representation to maximize both independent factorial axes.

## Do

### R7.1 Create a frozen benchmark config

Example:

```text
configs/benchmarks/regime_discovery_v1.yaml
```

Specify:

- case01–09 training grid;
- case10/case15 held-out cases;
- `bulk` pressure diagnostic;
- `rms_profile` thermal diagnostic;
- MFA combined sanity check;
- expected metric ranges.

### R7.2 Add regression tests

Not full expensive runs in CI.

Use reduced/synthetic or cached feature artifacts and assert that:

- pressure specialist remains the expected best block;
- thermal specialist remains the expected best block;
- MFA preserves expected OOD pressure assignment;
- deterministic outputs do not drift unexpectedly.

### R7.3 Produce one command

```bash
metis benchmark regime-v1
```

which reproduces the current result tables/JSON.

## Why this matters

The first scientific experiment becomes a **platform acceptance test**.

That is exactly how mature scientific software should evolve.

---

# 13. Milestone I1 — Experiment tracking with MLflow

**Priority:** high after R1–R7  
**Type:** new implementation  
**Career value:** very high

`mlflow` is already listed in the optional `ml` dependencies but is not yet integrated.

## Goal

Every ML/discovery experiment should generate a reproducible run.

## Track

- experiment name;
- dataset artifact ID;
- model/config;
- parameters;
- random seed;
- metrics;
- feature lists;
- figures;
- serialized model;
- Git commit;
- runtime;
- hardware where relevant.

## Start with one workflow

Do not MLflow-enable everything at once.

First integrate it into the **next new ML-discovery model**.

Then migrate useful existing regime experiments if worthwhile.

## Local-first setup

Use:

```text
mlruns/
```

initially.

No server/database deployment is needed in v1.

## Acceptance criteria

Given an MLflow run ID, another user can determine:

- exactly what data were used;
- how the model was configured;
- what metrics were produced;
- where the artifacts are.

---

# 14. Milestone I2 — First genuinely new ML discovery model

**Priority:** high  
**Type:** scientific + ML implementation

This is the next major scientific implementation after the refactors.

## Recommended first model

**Autoencoder-based representation learning across DNS cases.**

Why:

- directly supports the project's physical-discovery identity;
- extends beyond the already-settled PCA/clustering reference task;
- provides a nonlinear counterpart to POD/PCA;
- can be compared against known modal structure;
- creates real train/validation/model-lifecycle infrastructure;
- is computationally manageable before full neural operators.

## Scientific question

> Does a nonlinear learned representation recover physically meaningful thermodynamic/dynamical structure that is not already captured by linear POD/PCA, and is that representation robust to held-out operating conditions?

## Inputs

Start with a controlled representation, not raw 3D fields.

Possible sequence:

### I2-A

RMS/profile-derived vectors.

### I2-B

2D slice fields.

### I2-C

Multi-variable slices if I2-B provides useful evidence.

## Baselines

Must include:

- PCA;
- POD where representation is field-based;
- current axis-specific diagnostics.

## Evaluation

Not only reconstruction error.

Evaluate:

- latent organization versus `Pb_Pc`;
- latent organization versus thermal forcing;
- held-out case placement;
- sensitivity to seeds;
- reconstruction of physically meaningful statistics;
- latent/physical correlation;
- whether learned features provide information unavailable to existing diagnostics.

## Stop criterion

If the autoencoder only reproduces PCA/POD structure with no additional interpretability or robustness, record that result and stop.

Do not escalate architecture complexity automatically.

---

# 15. Milestone I3 — Training framework

**Priority:** high, built alongside I2  
**Target package:** `metis.training`

## Goal

Avoid model-specific standalone training scripts.

## Required capabilities

- reproducible seeds;
- train/validation splits;
- optimizer configuration;
- early stopping;
- checkpointing;
- metric logging;
- CPU/GPU device selection;
- MLflow integration;
- resume support where useful.

## Keep the abstraction small

Do not build a full deep-learning framework.

A project-level trainer sufficient for:

- autoencoders;
- LSTM;
- selected PyTorch models

is enough.

## Acceptance criteria

A new model can be added without reimplementing:

- logging;
- checkpoint writing;
- split handling;
- experiment metadata.

---

# 16. Milestone I4 — Model interfaces

**Priority:** medium/high  
**Target package:** `metis.models`

Current subpackages are mostly placeholders:

- baselines
- classical_ml
- lstm
- fno
- wno
- deeponet

Do not fill all of them merely because directories exist.

## Refactor recommendation

Keep only model modules that correspond to actual experiments.

Implement a minimal protocol:

```python
class MetisModel(Protocol):
    def fit(...): ...
    def predict(...): ...
    def save(...): ...
    @classmethod
    def load(...): ...
```

For PyTorch representation models, a training-specific interface may be more appropriate than forcing sklearn semantics.

## First implemented models

Recommended order:

1. PCA baseline adapter;
2. simple autoencoder;
3. classical baseline if needed;
4. LSTM only when temporal forecasting starts;
5. FNO/DeepONet only when a concrete field-learning question exists;
6. WNO only if scientifically justified.

## Delete or keep placeholders?

Either:

- keep empty namespaces but mark them explicitly as planned; or
- remove premature directories and add them when required.

Avoid giving the impression that unimplemented models are supported.

---

# 17. Milestone I5 — Physics-aware evaluation API

**Priority:** high  
**Target package:** `metis.evaluation`

## Goal

Create a shared evaluation layer combining conventional ML metrics with physical validation.

## Evaluation categories

### Generic

- MAE;
- RMSE;
- normalized RMSE;
- relative L2;
- R² where appropriate.

### Representation

- explained/reconstructed variance;
- latent stability across seeds;
- latent-to-physical-variable correlations;
- cluster/class separability only where structurally meaningful.

### Physical

Depending on model/output:

- mean profiles;
- RMS profiles;
- spectra;
- POD modal energy;
- reconstructed modes;
- derived dimensionless quantities;
- wall quantities.

### OOD

- held-out case performance;
- domain-distance diagnostics;
- degradation relative to in-distribution performance.

## Design rule

A model cannot be described as "better" only because one generic ML metric improves.

The relevant physical diagnostics must agree.

---

# 18. Milestone I6 — Reporting layer

**Priority:** medium  
**Type:** new implementation

## Goal

Generate consistent outputs from an experiment without manually rebuilding plots/tables.

Example:

```bash
metis report --run-id <id>
```

Outputs:

```text
reports/<run_id>/
    summary.md
    metrics.json
    figures/
        ...
```

## First report types

- case physics summary;
- regime benchmark report;
- representation-learning report;
- OOD comparison report.

## Why this matters

For the group:

- easier student supervision;
- consistent comparisons;
- reproducible figures.

For publications:

- less duplicated plotting code;
- traceable figure provenance.

---

# 19. Milestone I7 — CLI / user-facing command layer

**Priority:** medium  
**Type:** platform implementation

Do this after core APIs stabilize.

## Desired commands

```bash
metis cases list
metis case validate case01

metis analyze physics case01
metis analyze spectra case01
metis analyze pod case01

metis dataset build --config ...
metis train --config ...
metis evaluate --run-id ...
metis report --run-id ...

metis benchmark regime-v1
```

Use `argparse`, `typer`, or another lightweight CLI tool.

The exact library is less important than:

- discoverability;
- good help text;
- meaningful errors.

---

# 20. Milestone I8 — Registry and artifact management

**Priority:** medium  
**Target package:** `metis.registry`

## Goal

Track validated reusable models and datasets without creating enterprise-grade infrastructure.

## Minimal registry

A directory-based registry is sufficient:

```text
artifacts/
    datasets/
    models/
    reports/
    registry.json
```

Each model entry:

```json
{
  "model_id": "ae_regime_v1",
  "run_id": "...",
  "dataset_id": "regime_slices_v2",
  "status": "validated",
  "created_at": "...",
  "git_commit": "...",
  "metrics": {
    "...": "..."
  }
}
```

Possible statuses:

- experimental;
- validated;
- deprecated.

## Key rule

"Validated" means:

> approved for reuse under a documented scope.

It does not mean commercially deployed.

---

# 21. Milestone I9 — Temporal / virtual-sensor module

**Priority:** medium after representation-learning track  
**Scientific value:** potentially high  
**Career value:** very high

This should be the next major application track after the ML-discovery framework works.

## Goal

Create an industrial-style time-series task from DNS data without pretending it is physical sensor data.

Use **virtual probes extracted from DNS**.

## Work required

### Data

- define sensor locations;
- define variables;
- generate temporal windows;
- ensure physical timestep availability before claiming frequency/time units;
- if only iteration counts exist, preserve that limitation explicitly.

### Baselines

- persistence;
- autoregressive model;
- lagged linear regression;
- tree-based model.

### ML

- LSTM;
- temporal CNN if useful.

### Evaluation

- leakage-safe temporal splits;
- forecast horizon;
- OOD across operating conditions;
- physical/statistical consistency.

## Important limitation from current data

The current slice product does not store physical timestep alongside snapshots, only solver iteration numbers.

Therefore:

- do not claim physical-frequency temporal analysis from this source;
- either recover/validate the physical timestep from authoritative metadata;
- or formulate sequence forecasting in iteration/sample-step units.

---

# 22. Milestone I10 — Full-field neural models

**Priority:** later  
**Type:** advanced research

Only start after:

- dataset abstraction exists;
- training infrastructure exists;
- MLflow exists;
- evaluation exists;
- a specific scientific question is frozen.

Candidate models:

- convolutional autoencoder;
- FNO;
- DeepONet;
- U-Net variants;
- WNO if justified.

## Scientific use cases

- field compression;
- reconstruction;
- future-state prediction;
- cross-condition representation learning;
- learned ROM comparison.

## Do not

Build every architecture simply to populate `models/`.

Architecture count is not a scientific contribution.

---

# 23. Milestone P1 — Packaging cleanup

**Priority:** medium  
**Type:** professionalization

## Tasks

- add semantic project versioning;
- ensure clean install from fresh environment;
- expose CLI entry point;
- remove generated `*.egg-info` from tracked source if present;
- add package-level public API where helpful;
- add dependency groups deliberately;
- document Python version;
- consider lockfile/environment specification for exact reproducibility.

## Suggested extras

```text
dev
ml
router
demo
docs
```

Keep core installation lightweight.

---

# 24. Milestone P2 — Test strategy expansion

**Priority:** continuous

Current tests are a strong starting point.

Expand into three layers.

## Unit

Fast, no real group data.

Test:

- readers with mock fixtures;
- validation;
- preprocessing;
- metrics;
- model components;
- registry logic.

## Integration

Small synthetic/reduced datasets.

Test:

```text
ingest
→ validate
→ preprocess
→ build dataset
→ train tiny model
→ evaluate
→ save artifact
```

## Regression

Protect known scientific results.

Examples:

- published case01 physics values;
- spectra Parseval consistency;
- POD energy/rank;
- regime-discovery benchmark metrics.

## CI constraint

Do not require:

- private DNS data;
- GPU;
- external local repositories;
- API credentials

for the default CI job.

Optional jobs may cover these environments separately.

---

# 25. Milestone P3 — CI/CD expansion

**Priority:** medium after training exists

Current CI:

```text
install
→ Ruff
→ pytest
```

Keep this.

Later add:

```text
package build
→ install built wheel
→ smoke test CLI
```

Potential optional jobs:

- ML extra;
- router extra where dependency access permits;
- documentation build.

Avoid turning CI into a complex deployment system prematurely.

---

# 26. Milestone P4 — Docker

**Priority:** later but worthwhile  
**Career value:** high  
**Scientific necessity:** moderate

## First container target

Do **not** containerize HPC training first.

Containerize a lightweight reproducible analysis:

```text
mock/reduced dataset
→ physics/features
→ validated model/artifact
→ report
```

or later an inference/API service.

## Acceptance criteria

A new user can run:

```bash
docker compose up
```

or a documented `docker run` command and reproduce one complete demonstration workflow.

---

# 27. Milestone P5 — API

**Priority:** later  
**Target package:** `metis.api`

The API directory is currently empty.

Only implement after a stable reusable operation exists.

Good first endpoints:

```text
GET  /health
GET  /cases
GET  /analyses
POST /analyze
GET  /runs/{id}
```

Later, for trained models:

```text
POST /predict
POST /ood-check
```

## Important

The API is primarily:

- a standardized access layer;
- portfolio evidence;
- a future integration point.

It should not drive the scientific architecture.

---

# 28. Milestone P6 — Monitoring

**Priority:** later  
**Target package:** `metis.monitoring`

Separate two meanings.

## Software monitoring

- run failures;
- API errors;
- latency;
- model version;
- logs.

## Scientific monitoring

Much more interesting for METIS:

- distribution shift;
- latent-space shift;
- physical-consistency failure;
- model uncertainty;
- expected reliability.

Implement scientific monitoring only after there is a trained model whose validity domain is understood.

---

# 29. Milestone P7 — Cloud demonstration

**Priority:** late / optional  
**Career value:** high  
**Group value:** limited initially

Do not move the DNS archive to cloud merely to claim cloud experience.

Deploy a reduced demonstration workflow.

Possible target:

- Azure, given recurring relevance in job postings;
- alternatively AWS/GCP if group infrastructure makes one easier.

Minimum useful demonstration:

```text
validated small model
+ container
+ API
+ remote deployment
+ logs
+ documented setup
```

This is enough to truthfully claim practical cloud deployment experience.

---

# 30. Router module refactoring

The router is already feature-complete for its intended portfolio scope.

Do not allow it to consume Track A/B development time.

## Required cleanup

### Router-R1 — Remove filesystem coupling

Replace:

- absolute `PUB5_ROOT`;
- absolute package dependency.

Create an explicit adapter/config interface.

### Router-R2 — Freeze validated policy

Do not casually change the OOD routing rule.

Any routing-policy modification should require:

- new evidence;
- updated tests;
- updated documentation.

### Router-R3 — Keep deterministic decision outside LLM

Preserve the current good design:

```text
LLM explains
Python policy routes
```

Do not move routing control into free-form agent reasoning.

### Router-R4 — Live LLM test remains optional

The only current known missing end-to-end check is the live LLM round trip requiring credentials.

This should not block the main METIS framework.

---

# 31. Documentation refactor

**Priority:** high

The repository already has valuable documentation, but it can become easier for a new student to navigate.

## Recommended structure

```text
README.md
docs/
    getting_started.md
    architecture.md
    data_layout.md
    adding_an_analysis.md
    adding_a_model.md
    reproducibility.md
    research/
        regime_discovery.md
    router/
        architecture.md
        demo.md
```

## README should answer only

1. What is METIS?
2. How does it relate to RHEA?
3. What can it currently do?
4. How do I install it?
5. How do I run one reproducible example?
6. Where do I read more?

Move long implementation history out of README.

## Student onboarding target

A new student should be able to:

- install METIS;
- generate mock data;
- run tests;
- run one analysis;
- understand where to add a new module

within the first working session.

---

# 32. Proposed directory structure after refactor

```text
metis/
├── configs/
│   ├── default.yaml
│   ├── cases/
│   ├── datasets/
│   ├── experiments/
│   └── benchmarks/
│
├── src/metis/
│   ├── config/
│   ├── data/
│   │   ├── ingestion/
│   │   ├── validation/
│   │   ├── preprocessing/
│   │   ├── datasets/
│   │   └── registry.py
│   │
│   ├── features/
│   │   ├── physics.py
│   │   ├── spectra.py
│   │   ├── pod.py
│   │   └── regime.py
│   │
│   ├── models/
│   │   ├── baselines/
│   │   └── autoencoder/
│   │
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   ├── registry/
│   ├── reporting/
│   ├── monitoring/
│   ├── api/
│   ├── router/
│   ├── cli/
│   └── testing/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── artifacts/            # gitignored
│   ├── datasets/
│   ├── models/
│   └── reports/
│
├── scripts/              # thin transitional wrappers only
├── docs/
├── research_protocol.md
└── FINDINGS.md
```

Do not restructure everything in one commit.

Move incrementally while tests remain green.

---

# 33. Recommended execution order

## Stage 1 — Make current science reusable

### 1. Configuration/path refactor
R1

### 2. Case registry
R2

### 3. Validation
R3

### 4. Preprocessing
R4

### 5. Dataset/artifact abstraction
R5

### 6. Common analysis interface
R6

### 7. Freeze regime benchmark
R7

**Outcome:**

A second researcher can reproduce existing validated analyses without editing source code.

This is the first major group-level milestone.

---

## Stage 2 — Build the real ML experimentation layer

### 8. MLflow
I1

### 9. Autoencoder/representation-learning reference study
I2

### 10. Training framework
I3

### 11. Minimal model interface
I4

### 12. Physics-aware evaluation
I5

### 13. Reporting
I6

**Outcome:**

METIS is no longer only a physics post-processing package. It becomes a real ML research framework with tracked experiments and interpretable physical evaluation.

---

## Stage 3 — Make the platform easy for others

### 14. CLI
I7

### 15. Dataset/model registry
I8

### 16. Documentation cleanup

### 17. Expanded integration/regression tests

### 18. Packaging cleanup
P1/P2/P3

**Outcome:**

A student can perform a study through documented interfaces rather than reading the original author's scripts.

---

## Stage 4 — Add an industry-relevant temporal track

### 19. Virtual-sensor dataset
I9

### 20. Forecasting baselines

### 21. LSTM / temporal model

### 22. OOD evaluation across physical cases

**Outcome:**

METIS demonstrates a conventional industrial ML workflow while remaining scientifically grounded in DNS data.

---

## Stage 5 — Advanced models only when justified

### 23. Full-field learning
I10

### 24. Neural operators

### 25. Model trust / OOD monitoring

**Outcome:**

Advanced Scientific ML capabilities extend a stable platform rather than being isolated research scripts.

---

## Stage 6 — Production-style demonstration

### 26. Docker
P4

### 27. API
P5

### 28. Monitoring
P6

### 29. Cloud deployment
P7

**Outcome:**

The repository demonstrates the missing engineering lifecycle relevant to industry roles:

```text
data
→ validation
→ preprocessing
→ model
→ experiment tracking
→ testing
→ artifact
→ API
→ container
→ remote deployment
→ monitoring
```

---

# 34. Suggested first 8 weeks

This schedule assumes the focus is Track A/B platform development rather than the already-shipped router.

## Week 1

- introduce configuration loader;
- remove hard-coded regime-script data roots;
- define case metadata schema;
- create `CaseRegistry`.

## Week 2

- implement structural validation;
- implement numerical validation;
- add corrupted mock fixtures;
- add validation tests.

## Week 3

- implement preprocessing primitives;
- define dataset artifact metadata;
- extract/cached regime feature dataset;
- make current regime workflow consume cached features.

## Week 4

- freeze `regime-v1` benchmark;
- add regression tests;
- expose one reproducible benchmark command;
- update documentation.

### Gate 1

At the end of Week 4:

> Another machine/user should be able to reproduce the reference task without changing Python code.

Do not proceed if this is not true.

## Week 5

- integrate MLflow local tracking;
- define experiment abstraction;
- define autoencoder research protocol;
- prepare first training dataset.

## Week 6

- implement simple autoencoder;
- implement PCA baseline through same evaluation pipeline;
- add training/checkpoint infrastructure;
- log experiments to MLflow.

## Week 7

- run seed/sensitivity experiments;
- evaluate latent organization versus physical operating conditions;
- implement physics-aware representation diagnostics.

## Week 8

- analyze whether nonlinear representation adds information beyond PCA/POD;
- write results into `FINDINGS.md`;
- decide:
  - continue representation-learning paper;
  - modify hypothesis;
  - stop and move to temporal track.

### Gate 2

Do not add FNO/WNO/DeepONet merely because Week 8 is reached.

Advanced models require a result-driven justification.

---

# 35. Backlog classification

## Do now

- remove hard-coded paths;
- configuration system;
- case registry;
- validation layer;
- preprocessing layer;
- dataset artifacts;
- freeze regime benchmark;
- regression tests;
- MLflow;
- first nonlinear representation model;
- physics-aware evaluation;
- documentation.

## Do next

- common CLI;
- experiment/reporting layer;
- simple registry;
- virtual-sensor forecasting;
- LSTM/classical temporal baselines;
- packaging improvements.

## Do later

- neural operators inside METIS;
- API;
- Docker;
- monitoring;
- cloud deployment.

## Explicitly defer unless a research need appears

- Kubernetes;
- Kafka;
- Spark;
- Databricks;
- Airflow;
- Terraform;
- microservices;
- distributed training platform;
- live Slurm submission/orchestration;
- large web frontend.

These technologies should not be added for CV keyword collection.

---

# 36. Definition of “group-ready v1”

METIS should be considered **group-ready v1** when all of the following are true:

## Installation

A new user can:

```bash
git clone ...
pip install -e ".[dev]"
```

without modifying package files.

## Data setup

The user can point METIS at the group's DNS data through configuration.

## Validation

The framework can report whether a case is structurally suitable for a requested analysis.

## Standard physics

The user can reproduce:

- case physics summary;
- spectra;
- POD;
- regime features.

## Benchmark

The user can reproduce the frozen regime-discovery benchmark.

## Reproducibility

Results record:

- configuration;
- data/case IDs;
- seed;
- Git commit;
- output artifacts.

## Testing

Core CI passes without access to private DNS data.

## Documentation

A new student can complete a tutorial workflow without author assistance.

---

# 37. Definition of “ML-platform v1”

After group-ready v1, the stronger milestone is **ML-platform v1**.

Requirements:

- reusable dataset artifact;
- train/validation/OOD split management;
- at least one trained ML model;
- PCA/POD or equivalent baseline;
- MLflow experiment tracking;
- checkpoint/model artifact;
- physics-aware evaluation;
- regression/integration tests;
- reproducible report;
- documented model validity scope.

At that point METIS is credibly an ML research platform rather than only a post-processing toolkit.

---

# 38. Definition of “portfolio production v1”

This milestone targets the recurring industry gap.

Requirements:

- everything in ML-platform v1;
- clean package installation;
- CI;
- Docker;
- FastAPI or equivalent service;
- model/artifact loading;
- basic logging;
- one remote cloud deployment;
- simple monitoring;
- README architecture diagram;
- one end-to-end example.

Then it becomes accurate to state:

> Built a reusable ML platform for high-dimensional DNS data covering validated ingestion, automated physical feature extraction, dataset generation, experiment tracking, model training, OOD evaluation, physics-aware validation, testing, CI/CD, containerized inference, and cloud deployment.

---

# 39. Scientific guardrails

METIS should preserve the methodological standards already demonstrated by the regime study.

## 39.1 No architecture-first research

Start from a physical question.

Do not start from:

> “We should use a transformer.”

## 39.2 Negative results are results

The existing finding that one clustering cannot reproduce two orthogonal factorial partitions is valuable.

Do not hide it by endless model tuning.

## 39.3 No test leakage

OOD cases and held-out regimes must remain genuinely unseen during model/preprocessing fitting when the task claims generalization.

## 39.4 Compare learned representations to known physics

Latent spaces require interpretation.

## 39.5 DNS remains authoritative

METIS extracts or models information from DNS.

It does not redefine the reference physics.

## 39.6 Avoid unsupported temporal claims

If physical timestep information is unavailable in a particular data product, report sequence position/iteration rather than inventing physical frequency.

---

# 40. Software guardrails

## 40.1 No absolute paths in source

Environment-specific paths belong in config/environment variables.

## 40.2 No private data required for core CI

Use mocks/reduced fixtures.

## 40.3 Do not over-abstract early

Interfaces should emerge from at least two real use cases.

## 40.4 Do not duplicate validated methods

Wrap existing validated physics code rather than rewriting it unnecessarily.

## 40.5 Every model has a validity scope

Record:

- training cases;
- variables;
- preprocessing;
- intended operating domain;
- known OOD behavior.

## 40.6 Every artifact should be traceable

A figure or metric should be connectable back to an experiment/configuration.

---

# 41. Recommended next commit sequence

Keep changes reviewable.

## Commit 1

**`refactor: add configurable data root`**

- config loader;
- CLI flags;
- remove hard-coded `DATA_ROOT`.

## Commit 2

**`feat: add case registry and descriptors`**

- centralized metadata;
- tests.

## Commit 3

**`feat: add DNS validation layer`**

- structural/numerical checks;
- corrupted fixtures.

## Commit 4

**`feat: add reusable preprocessing transforms`**

- normalization;
- variable selection;
- serialization tests.

## Commit 5

**`feat: add dataset artifacts and caching`**

- extracted feature dataset;
- metadata.

## Commit 6

**`refactor: run regime benchmark from cached dataset`**

- same science;
- cleaner execution.

## Commit 7

**`test: freeze regime-v1 scientific regression benchmark`**

- expected metrics/artifacts.

## Commit 8

**`feat: integrate MLflow experiment tracking`**

## Commit 9

**`feat: add autoencoder representation baseline`**

## Commit 10

**`feat: add physics-aware representation evaluation`**

This order minimizes the chance of breaking trusted existing functionality.

---

# 42. What not to refactor yet

Several things are tempting but low-return now.

Do not spend time on:

- converting every function into a class;
- redesigning the entire package namespace;
- building a plugin system;
- adding database infrastructure;
- rewriting POD/spectra implementations that already match reference results;
- adding async APIs;
- sophisticated logging frameworks;
- distributed compute abstractions;
- generic workflow engines.

The current scientific core works.

The goal is to **make it reusable and extensible**, not architecturally fashionable.

---

# 43. Best next scientific implementation

After the infrastructure refactor, the recommended first new scientific capability is:

> **Nonlinear representation learning of transcritical DNS structure, benchmarked against PCA/POD and interpreted through known pressure/thermal physics.**

This is preferred over immediately implementing forecasting because it:

- aligns most directly with the stated group objective;
- exploits the existing regime-discovery baseline;
- provides a publishable continuation;
- introduces real model training;
- requires the MLOps/reproducibility infrastructure;
- creates a natural bridge to later autoencoders/neural operators.

The virtual-sensor forecasting track should follow as the strongest industry-facing second application.

---

# 44. Best next engineering implementation

The single most valuable engineering sequence is:

```text
configuration
→ dataset artifact
→ MLflow
→ trainable model
→ physics-aware evaluation
→ tests
```

Not:

```text
FastAPI
→ Docker
→ cloud
```

The deployment layer is valuable only once there is a clean, reproducible model artifact worth deploying.

---

# 45. Final priority table

| Priority | Item | Type | Main value |
|---|---|---|---|
| 1 | Remove hard-coded paths/configure data | Refactor | Portability |
| 2 | Case registry | Refactor | Single source of truth |
| 3 | Validation layer | Implement | Reliability |
| 4 | Preprocessing layer | Implement | Reuse + leakage control |
| 5 | Dataset artifacts/cache | Implement | Experiment speed/reproducibility |
| 6 | Freeze regime-v1 benchmark | Consolidate | Scientific regression test |
| 7 | MLflow | Implement | Experiment lifecycle |
| 8 | Autoencoder representation study | Implement | New scientific ML |
| 9 | Training framework | Implement | Reusable model development |
| 10 | Physics-aware evaluation | Implement | Scientific credibility |
| 11 | Reporting/CLI | Implement | Student usability |
| 12 | Registry | Implement | Reusable validated models |
| 13 | Virtual-sensor forecasting | Implement | Industry-relevant ML |
| 14 | Full-field neural models | Research | Advanced Scientific ML |
| 15 | Docker/API | Engineering | Production-style evidence |
| 16 | Monitoring | Engineering | Model lifecycle |
| 17 | Cloud demo | Engineering | Close cloud experience gap |

---

# 46. Immediate next action

Start with **Milestone R1**.

Before implementing any new model, make this command conceptually possible from a clean checkout:

```bash
metis benchmark regime-v1 \
    --data-root /path/to/group/data
```

It should:

1. discover the requested cases;
2. validate their metadata/data products;
3. load or generate the required features;
4. reproduce the frozen regime benchmark;
5. write a versioned result artifact;
6. require no source-code edits.

Once that works, METIS has crossed an important boundary:

> from *Guillem's research repository* to *a research tool another person can actually use*.

That is the correct foundation for every future ML, publication, student project, and production-style engineering layer.

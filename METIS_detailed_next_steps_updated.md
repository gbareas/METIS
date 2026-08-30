# METIS — Updated Detailed Next Steps

**Status baseline:** repository snapshot reviewed 2026-08-30  
**Project:** METIS — Machine-learning Engine for Transcritical Insight & Statistics  
**Core identity:** **RHEA generates the physics; METIS extracts the insight.**

---

# 1. Purpose

This document is the current execution plan for METIS.

It supersedes the earlier roadmap where configuration, validation, preprocessing, dataset abstraction, ML tracking, model training, reporting, CLI, and registry were still mostly future work.

The repository has now crossed an important boundary:

> **METIS is no longer only reusable scientific post-processing code. It is an early but credible ML experimentation platform.**

The priority is therefore no longer to keep generalizing the software architecture.

The priority is now:

1. **correct the remaining scientific/reproducibility issues;**
2. **use the platform for a richer ML research problem;**
3. **add an industry-recognizable temporal ML workflow;**
4. **take one validated model through deployment, monitoring, and cloud;**
5. **only then expand into more advanced architectures where scientifically justified.**

The project should continue to optimize simultaneously for:

- publishable scientific value;
- reuse by the research group and future students;
- industrial ML/software-engineering relevance.

---

# 2. Current state

## 2.1 High-level assessment

Current METIS can reasonably be described as:

> **A tested Python framework for ingesting and validating high-fidelity DNS data, extracting reproducible physical features, benchmarking data-driven representations, tracking ML experiments, training models, evaluating them against physically meaningful diagnostics, and managing reproducible artifacts.**

It should **not yet** be described as:

- production ML;
- a cloud ML platform;
- a deployed inference system;
- an MLOps production stack.

Those are the next professionalization layers.

---

# 3. Current capability scorecard

| Area | Current status |
|---|---:|
| Scientific validity | **Strong** |
| DNS ingestion / physics integration | **Strong** |
| Testing | **Very strong** |
| Portability | **Strong, with a few remaining debts** |
| Reproducibility | **Strong** |
| Group usability | **Good / improving** |
| ML experimentation | **Real and usable** |
| ML model lifecycle | **Early but credible** |
| Production/MLOps | **Still limited** |
| Cloud | **Not yet demonstrated** |
| Monitoring | **Not yet demonstrated** |
| SQL/data-platform exposure | **Still limited** |

Latest reviewed self-contained test suite:

```text
209 passed
4 skipped
```

This is sufficient evidence that the recent work is architectural and functional, not only documentation.

---

# 4. What is already implemented

The following should now be treated as existing platform capabilities rather than future roadmap items.

## 4.1 DNS ingestion

Implemented:

- native RHEA HDF5 ingestion;
- ghost-cell handling;
- processed metadata integration;
- time-resolved slice ingestion;
- structured DNS case objects.

The ingestion layer has already demonstrated its value by exposing a real coordinate/axis-order defect during validation work.

That is an important engineering result:

> automated data validation is already preventing silent scientific errors.

---

# 5. Case registry and configuration

METIS now includes explicit case/configuration abstractions that remove much of the old script-level filesystem logic.

Existing direction:

```text
configuration
     ↓
CaseRegistry
     ↓
CaseDescriptor
     ↓
reader / analysis
```

This is the correct architecture and should be preserved.

No broad rewrite is needed.

---

# 6. Data validation

The validation layer is now substantive.

Current checks include areas such as:

- required fields;
- array dimensions;
- RHEA axis conventions;
- coordinate consistency;
- NaNs / infinities;
- physically invalid values;
- RMS constraints;
- duplicate frames;
- cross-case compatibility;
- slice consistency;
- timestep metadata.

This layer should now become the mandatory gate for shared group workflows.

---

# 7. Preprocessing

METIS now has explicit reusable preprocessing.

The important design rule is already present:

```text
fit on training data
        ↓
apply to validation / OOD
```

This is essential for any future claim of generalization.

Do not regress back to convenience preprocessing where held-out information can leak into training.

---

# 8. Dataset / feature artifact layer

The platform now separates expensive DNS processing from cheap repeated experimentation.

Target pattern:

```text
raw DNS
   ↓
validated extraction
   ↓
versioned dataset artifact
   ↓
many cheap experiments
```

This is one of the most important pieces for both:

- research reproducibility;
- industry-style ML architecture.

Keep extending this layer rather than forcing each model to reread raw DNS.

---

# 9. Physics-analysis layer

Trusted existing modules include:

- bulk / wall-normal physical statistics;
- spectra;
- POD;
- regime features;
- cross-case physical summaries.

These should continue to act as the **scientific baseline** against which learned representations are interpreted.

The goal of ML in METIS is not to replace these methods automatically.

The goal is to determine when ML provides additional physical information.

---

# 10. Frozen regime-discovery benchmark

The regime-discovery study should now be treated as a **reference scientific regression benchmark**.

Current conclusions remain important:

- pressure information is dominated by the `bulk` feature block;
- thermal information is dominated by `rms_profile`;
- `mean_profile` is not useful for the reference objective;
- POD-energy fractions do not improve that particular case-level task;
- one `k=3` clustering cannot simultaneously recover two independent 3-level factorial axes.

Do **not** keep tuning this task unless a new scientific hypothesis is introduced.

Its value now is:

```text
known scientific result
       ↓
code change
       ↓
regression benchmark
       ↓
did the scientific result change unexpectedly?
```

---

# 11. ML experiment tracking

METIS now includes experiment tracking.

The expected run provenance should include:

- dataset ID;
- dataset fingerprint;
- train cases;
- validation cases;
- OOD cases;
- preprocessing;
- architecture;
- hyperparameters;
- random seed;
- training history;
- evaluation metrics;
- figures;
- checkpoint/artifact;
- Git commit;
- runtime/hardware metadata where useful.

Local-first tracking is sufficient at this stage.

Do **not** deploy an MLflow server merely to make the architecture look more industrial.

---

# 12. Training framework

A reusable PyTorch training layer now exists.

This is the correct point to let real experiments define future abstractions.

Avoid building a universal framework.

The trainer should remain focused on shared requirements such as:

- deterministic/reproducible seeding;
- optimizer configuration;
- early stopping;
- checkpointing;
- CPU/GPU selection;
- MLflow logging;
- validation;
- resuming where useful.

---

# 13. Representation models

METIS now includes at least:

- PCA representation baseline;
- autoencoder representation model.

This is enough to support the first nonlinear representation study.

Do **not** populate all planned model namespaces simply because they exist.

New architectures should enter METIS only when tied to a research question.

---

# 14. Physics-aware evaluation

The evaluation layer is now moving beyond generic ML metrics.

This is one of METIS's strongest differentiators.

The evaluation philosophy should remain:

> A model is not better because one scalar ML metric improves.

A learned model should be evaluated through a combination of:

- reconstruction/prediction metrics;
- robustness;
- seed sensitivity;
- latent structure;
- operating-condition organization;
- POD/PCA comparison;
- physical statistics;
- spectra where relevant;
- held-out/OOD behavior.

---

# 15. Reporting

Reproducible report generation is now part of the platform.

Continue moving toward:

```bash
metis report --run-id <id>
```

producing:

```text
summary.md
metrics.json
figures/
provenance.json
```

This is valuable for:

- student supervision;
- paper figure provenance;
- experiment comparison;
- reproducibility.

---

# 16. CLI

The CLI is now a real part of the platform.

The intended user experience should continue toward:

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

The CLI should remain a thin interface over stable Python APIs.

Do not move business/scientific logic into CLI handlers.

---

# 17. Artifact registry

METIS now has an artifact/model lifecycle concept.

Recommended statuses:

```text
experimental
    ↓
validated
    ↓
deprecated
```

Here:

> **validated** = approved for reuse within a documented scientific validity domain.

It does not mean commercial production deployment.

Each validated model should eventually record:

- training cases;
- input variables;
- preprocessing;
- intended task;
- operating domain;
- known OOD limitations;
- reference run;
- Git commit;
- dataset fingerprint.

---

# 18. Immediate scientific correctness issue — autoencoder seed bug

## Priority

**P0 — fix before further interpretation of the current representation study.**

The current autoencoder construction resets PyTorch to seed `0` inside network creation.

Conceptually:

```python
torch.manual_seed(0)
model = Autoencoder(...)
```

This means multiple nominal experiment seeds can start from the same weights.

If the training process is otherwise deterministic, this invalidates claims based on:

- zero seed-to-seed variance;
- effectively identical latent spaces;
- claims that the nonlinear optimum is uniquely stable.

## Required fix

The experiment seed must be applied **before parameter initialization**.

Preferred pattern:

```python
torch.manual_seed(seed)
model = Autoencoder(...)
trainer = Trainer(seed=seed, ...)
```

Remove hard-coded initialization seeds from internal model builders unless deterministic initialization is explicitly requested.

## Required test

Add a test proving:

```text
seed A → initial parameters A
seed B → initial parameters B
```

and that:

```text
same seed → same initial parameters
different seed → at least one different parameter tensor
```

## Required research action

After fixing:

1. rerun the multi-seed representation study;
2. regenerate metrics;
3. regenerate latent-stability results;
4. update MLflow artifacts;
5. update `representation_study.json`;
6. correct `FINDINGS.md`.

## Expected interpretation

Previous diagnostic reruns suggest that the main conclusion likely survives:

> The case-level autoencoder does not reliably outperform PCA and is more initialization-sensitive.

That is a valid and useful negative result.

The wording must change from:

> “zero variance proves stability”

to something closer to:

> “Across genuinely independent initializations, the nonlinear model does not provide a robust improvement over the linear baseline.”

---

# 19. Immediate refactor 1 — persistent benchmark caching

## Priority

**P0/P1 — small fix, high value.**

The frozen `regime-v1` benchmark should reuse persisted feature artifacts.

The benchmark should not repeatedly regenerate feature blocks from raw DNS when nothing has changed.

Target:

```text
artifacts/
└── datasets/
    └── regime-v1/
        ├── bulk/
        ├── rms_profile/
        ├── mean_profile/
        └── pod/
```

Example behavior:

```bash
metis benchmark regime-v1
```

should:

1. resolve expected dataset fingerprint;
2. load matching cached artifact if available;
3. rebuild only when missing/invalid;
4. report whether cache was reused;
5. write benchmark output with provenance.

---

# 20. Immediate refactor 2 — stronger cache invalidation

Current feature fingerprints are not sufficient if they only represent:

- case IDs;
- feature configuration;
- selected feature-source code.

A change to:

- HDF5 reader;
- slice reader;
- metadata;
- raw source files;

can alter feature values without necessarily changing the current fingerprint.

## Recommended solution

Do **not** hash TB-scale datasets byte-by-byte.

Create a lightweight source manifest.

For each source artifact record:

```text
relative path
file size
modification timestamp
small authoritative metadata hash
```

Also include software provenance for the relevant ingestion/preprocessing layer.

Conceptually:

```text
dataset fingerprint =
    source manifest
  + ingestion version/hash
  + preprocessing config/hash
  + feature extraction version/hash
  + case IDs
```

The exact hashing implementation can remain lightweight.

The objective is reliable invalidation, not cryptographic archival integrity.

---

# 21. Immediate refactor 3 — `CaseRegistry.from_config()` consistency

## Priority

**P1 — small API correctness issue.**

When a Python mapping is passed directly:

```python
CaseRegistry.from_config(config=cfg)
```

the `paths` block should not be read from that mapping while `data.root` is resolved through a separate path that ignores the supplied mapping.

Required behavior:

```python
cfg = {
    "data": {"root": "/foo"},
    "paths": {...}
}
```

must produce a registry rooted at:

```text
/foo
```

unless an explicit CLI/function argument overrides it.

Recommended resolution order:

```text
explicit argument
    >
supplied config mapping
    >
config file
    >
environment variable
    >
clear error
```

Add a direct test for config-mapping root resolution.

---

# 22. Immediate refactor 4 — strict validation policy

## Priority

**P1 — important for group reuse.**

Current analyses should not quietly proceed after validation errors.

Recommended default:

```text
validation ERROR
      ↓
abort analysis

validation WARNING
      ↓
continue + record warning
```

Provide an expert override:

```bash
--allow-invalid
```

or:

```python
run_analysis(..., strict_validation=False)
```

This is the safer default for student/group usage.

No published/scientific artifact should be generated from known-invalid input without an explicit override recorded in provenance.

---

# 23. Stop broad architecture refactoring after these items

After Sections 18–22 are resolved:

> **stop generic refactoring.**

Do not spend another development cycle on:

- generic class hierarchies;
- abstract plugin frameworks;
- workflow engines;
- database systems;
- distributed execution;
- microservices;
- package reshuffling.

The current infrastructure is sufficient to support the next scientific experiment.

From that point onward, new abstractions should be created only when a real use case requires them.

---

# 24. Priority scientific track 1 — I2-B richer representation learning

## Priority

**P1 — next major scientific/ML implementation.**

The case-level autoencoder study used effectively:

```text
one feature vector per DNS case
```

with very small sample count.

That is an unfavorable setting for nonlinear representation learning.

The logical next question is:

> **Does nonlinear representation learning add physically meaningful information when trained on a much richer sample space?**

Recommended input:

**2D DNS slice fields** or another high-sample spatial representation.

---

# 25. I2-B scientific hypothesis

A useful falsifiable hypothesis:

> **A nonlinear representation learned from spatial DNS samples captures physically meaningful transcritical-flow structure that is not represented as efficiently or robustly by a linear PCA/POD basis, particularly across changing thermodynamic conditions.**

## Falsification condition

If the nonlinear model:

- does not improve representation quality meaningfully;
- is less robust to held-out cases;
- produces latent variables with no additional physical interpretation;
- is strongly seed-sensitive;
- or merely reproduces POD/PCA structure;

then the added complexity is not justified.

Record the negative result and stop.

---

# 26. I2-B recommended experiment design

## Data unit

Prefer many samples:

```text
case
  ↓
time-resolved 2D slices
  ↓
N spatial samples
```

rather than:

```text
case
  ↓
one vector
```

## Start with one variable

Example:

- `u`;
- temperature;
- another physically justified field.

Do not begin with every variable simultaneously.

## Baseline

Mandatory:

- PCA or POD.

## Nonlinear model

Start with:

- simple convolutional autoencoder.

Do not jump directly to:

- VAE;
- transformer;
- FNO;
- WNO;
- DeepONet.

---

# 27. I2-B split design

Avoid random sample splitting across all cases if the goal is physical generalization.

Recommended structure:

## Training

Selected operating conditions.

## Validation

Held-out temporal/spatial samples from training cases.

## OOD test

Entire operating condition(s) excluded from training.

Possible axes:

- pressure holdout;
- wall-temperature holdout;
- strong/weak pseudo-boiling regime;
- cross-condition extrapolation.

Keep the OOD definition physically meaningful.

---

# 28. I2-B evaluation

## Generic representation metrics

- reconstruction MAE/RMSE;
- relative L2;
- explained/reconstructed variance;
- compression ratio.

## Robustness

- multiple real initialization seeds;
- variance across seeds;
- held-out case performance;
- degradation under OOD shift.

## Physical interpretation

Compare latent variables against:

- pressure ratio;
- wall-temperature parameters;
- pseudo-boiling indicators;
- RMS quantities;
- modal energies;
- spectral features;
- physically meaningful derived quantities.

## Linear comparison

Compare directly with PCA/POD at equivalent latent dimension.

Questions:

- Does nonlinear compression need fewer coordinates?
- Are nonlinear latent coordinates more physically informative?
- Do they generalize better?
- Are they more stable?
- Do reconstructed fields preserve statistics/spectra better?

---

# 29. I2-B acceptance criteria

The study is successful if **either**:

### Positive result

The nonlinear representation provides a clear, reproducible benefit over PCA/POD.

or:

### Valuable negative result

The experiment shows convincingly that linear representations remain sufficient for this class of flow structure under the tested conditions.

Both are scientifically acceptable.

The platform should support stopping a method when evidence says it is unnecessary.

---

# 30. Priority industry track — virtual-sensor temporal modelling

## Priority

**P1/P2 — begin after I2-B is stable or in parallel if resources permit.**

This is the most important next application for reducing the academia→industry gap.

The project should transform DNS output into a conventional industrial temporal-ML problem:

```text
DNS fields
    ↓
virtual probes
    ↓
multivariate temporal signals
    ↓
windowed dataset
    ↓
forecasting / anomaly task
```

This creates directly recognizable experience in:

- time series;
- forecasting;
- sensor data;
- predictive modelling;
- temporal validation;
- distribution shift;
- anomaly detection;
- model monitoring.

---

# 31. Virtual-sensor data design

Choose physically meaningful probe locations.

Examples:

- cold-wall viscous/buffer region;
- cold-wall outer region;
- channel centre;
- hot-wall outer region;
- hot-wall buffer/viscous region.

Candidate variables:

- `u`;
- `v`;
- `w`;
- `T`;
- `rho`;
- `cp`;
- wall-related quantities where available;
- carefully selected derived observables.

Avoid generating hundreds of arbitrary sensors without a physical hypothesis.

---

# 32. Temporal task 1 — forecasting

Example formulation:

\[
X_{t-L:t}
\rightarrow
X_{t+1:t+H}
\]

where:

- `L` = context length;
- `H` = forecast horizon.

## Baselines

Mandatory:

1. persistence;
2. autoregressive baseline;
3. lagged linear regression.

## Classical ML

Recommended:

- XGBoost / LightGBM with lag/statistical features.

## Deep learning

Then:

- LSTM;
- temporal CNN if useful.

Only add transformer-style models if the simpler models demonstrate a specific limitation.

---

# 33. Temporal task 2 — anomaly/regime detection

Optional second problem:

> Can a model detect changes in thermodynamic/flow regime from virtual-sensor signals?

Possible formulations:

- supervised regime classification;
- unsupervised anomaly detection;
- change-point detection;
- OOD scoring.

This is especially useful if it connects temporal data to the existing regime-discovery work.

---

# 34. Temporal validation

This is critical for industry relevance.

Use:

- chronological splits;
- no future leakage;
- held-out operating conditions;
- OOD cases;
- forecast-horizon analysis.

Avoid:

```text
randomly shuffle all timesteps
```

for forecasting claims.

---

# 35. Important temporal metadata limitation

Do not claim physical-frequency forecasting if the authoritative physical timestep is unavailable for a given slice product.

If only solver iteration/sample index exists:

- report sample-step forecasting;
- recover the physical timestep from authoritative metadata before converting to seconds/frequency.

Do not infer physical time without evidence.

---

# 36. Industry-relevant dataset persistence

The virtual-sensor track is the right point to introduce more conventional data handling.

Recommended outputs:

```text
artifacts/datasets/virtual_sensors_v1/
    signals.parquet
    windows.parquet / npz
    metadata.json
    splits.json
```

Consider using Parquet for tabular/time-series products.

This creates experience closer to standard data-science workflows than raw scientific HDF5 alone.

---

# 37. SQL — targeted addition

SQL remains one of the recurring profile gaps.

Do **not** force SQL into raw field processing where it is inappropriate.

Use it naturally for structured metadata and temporal experiments.

Possible SQLite/PostgreSQL tables:

```text
cases
sensors
experiments
model_runs
predictions
metrics
```

Example use cases:

- query all sensors for a case;
- query runs by operating regime;
- retrieve model metrics;
- compare OOD performance;
- store batch-prediction metadata.

Minimum useful objective:

Be comfortable with:

- joins;
- CTEs;
- aggregations;
- window functions;
- analytical queries.

The project should demonstrate SQL usage where relational structure genuinely exists.

---

# 38. Priority production track — one model all the way to deployment

## Priority

**P2 — after one temporal model is validated.**

This is the most important step for closing the remaining production-ML gap.

Do **not** productionize every METIS model.

Choose one.

Best candidate:

> **validated virtual-sensor forecasting model**

because its inputs/outputs are compact and API-friendly.

Target lifecycle:

```text
validated dataset
      ↓
training
      ↓
MLflow
      ↓
validated model artifact
      ↓
registry
      ↓
FastAPI
      ↓
Docker
      ↓
CI/CD
      ↓
cloud
      ↓
logging + monitoring
```

---

# 39. Inference package

Use the existing `metis.inference` direction only when a validated model exists.

Responsibilities:

- model loading;
- preprocessing loading;
- input validation;
- prediction;
- output formatting;
- model version identification;
- OOD/reliability score where supported.

Avoid training logic in inference modules.

---

# 40. FastAPI

Implement a small API.

Recommended first endpoints:

```text
GET  /health
GET  /models
GET  /models/{id}

POST /predict
POST /ood-check
```

Example prediction payload:

```json
{
  "model_id": "sensor_forecast_lstm_v1",
  "sensor_data": [...],
  "horizon": 20
}
```

The API is a demonstration of production-style access.

It should remain thin over `metis.inference`.

---

# 41. Docker

Containerize **inference first**, not HPC training.

Target:

```bash
docker build ...
docker run ...
```

or:

```bash
docker compose up
```

should expose:

```text
/health
/predict
```

with:

- pinned dependencies;
- selected model artifact;
- logging;
- tests/smoke checks.

---

# 42. CI/CD expansion

Current CI should remain focused and fast.

Target pipeline:

```text
commit / PR
    ↓
lint
    ↓
unit tests
    ↓
integration tests
    ↓
scientific regression tests
    ↓
package build
    ↓
Docker build
    ↓
API smoke test
```

Deployment can later be triggered only from:

- tagged release;
- main branch;
- manually approved workflow.

Avoid complex CD before the local container is stable.

---

# 43. Monitoring

Monitoring should have two layers.

## Engineering monitoring

Track:

- request count;
- failures;
- latency;
- model version;
- API health.

## ML monitoring

Track:

- input-distribution drift;
- feature shift;
- OOD score;
- prediction distribution;
- model error when reference data later become available.

## Scientific monitoring

Potentially distinctive METIS contribution:

- modal-space shift;
- latent-space distance;
- violation of physical constraints;
- degradation indicators tied to thermodynamic regime changes.

This is where industrial MLOps and publishable Scientific ML can overlap.

---

# 44. Cloud deployment

## Priority

**P2/P3 — after Docker/API works locally.**

Deploy a small demonstration service.

Given recurring relevance in the job market, **Azure** is a reasonable first choice.

Minimum useful cloud scope:

- containerized inference service;
- one validated model;
- remote request;
- logs;
- environment/secrets configuration;
- documented deployment;
- simple monitoring.

Do not upload the full DNS database.

Use:

- reduced data;
- model artifact;
- small demonstration payloads.

---

# 45. Advanced Scientific ML — later

Only after the previous tracks are working should METIS invest heavily in:

- FNO;
- WNO;
- DeepONet;
- U-Net variants;
- advanced neural operators.

These models should answer specific questions such as:

- field reconstruction;
- temporal field prediction;
- cross-condition generalization;
- learned reduced-order representation.

Do not create an architecture zoo.

---

# 46. What to defer

Explicitly defer unless a real requirement appears:

- Kubernetes;
- Kafka;
- Spark;
- Databricks;
- Airflow;
- Terraform;
- distributed training platforms;
- microservices;
- complex frontend;
- custom feature store;
- online feature serving.

These tools are not needed yet.

Adding them only for CV keywords would reduce project coherence.

---

# 47. Router module

The router remains a useful separate portfolio feature.

Keep the current strong design principle:

```text
deterministic Python policy
        ↓
routing decision

LLM
        ↓
explanation / interface
```

Do not move scientific/validity decisions into unconstrained LLM reasoning.

Required maintenance:

- keep external paths portable;
- keep router integration optional;
- do not let router dependencies contaminate core METIS;
- keep live LLM tests optional.

Track A/B should remain METIS's scientific identity.

---

# 48. Documentation priority

The documentation should now be reorganized around **users**, not implementation history.

Recommended:

```text
README.md

docs/
    getting_started.md
    architecture.md
    data_layout.md

    analyses/
        physics.md
        spectra.md
        pod.md
        regime_discovery.md

    ml/
        experiment_tracking.md
        representation_learning.md
        temporal_forecasting.md

    development/
        adding_an_analysis.md
        adding_a_model.md
        testing.md
        reproducibility.md

    deployment/
        inference.md
        docker.md
        cloud.md

    router/
        architecture.md
```

---

# 49. Group-use acceptance criterion

A new MSc/BSc student should eventually be able to:

```bash
git clone ...
pip install -e ".[dev]"
metis cases list
metis case validate case01
metis analyze pod case01 ...
metis benchmark regime-v1
```

without:

- editing Python source;
- knowing the original author's directory structure;
- understanding internal HDF5 quirks;
- rewriting analysis code.

---

# 50. Industry-relevance criterion

Every major new feature should ideally satisfy at least two of:

1. **scientifically useful;**
2. **useful to other group members;**
3. **industry-relevant engineering/ML capability.**

Examples:

| Work item | Science | Group | Industry |
|---|:---:|:---:|:---:|
| data validation | ✓ | ✓ | ✓ |
| dataset artifacts | ✓ | ✓ | ✓ |
| MLflow | partial | ✓ | ✓ |
| field autoencoder | ✓ | ✓ | ✓ |
| virtual-sensor forecasting | ✓ | ✓ | ✓ |
| SQL metadata/experiments | partial | ✓ | ✓ |
| Docker | — | ✓ | ✓ |
| FastAPI | — | possible | ✓ |
| OOD monitoring | ✓ | ✓ | ✓ |
| cloud deployment | — | low | ✓ |
| Kubernetes now | — | — | partial |

This should guide scope decisions.

---

# 51. Updated execution order

## Phase A — correctness and small debt cleanup

### A1
Fix autoencoder initialization/seeding.

### A2
Rerun I2-A.

### A3
Update findings and tracked artifacts.

### A4
Enable persistent `regime-v1` feature caching.

### A5
Strengthen dataset/cache fingerprints.

### A6
Fix `CaseRegistry.from_config()` mapping behavior.

### A7
Make validation errors blocking by default.

### Exit criterion

All known scientific/reproducibility issues are resolved.

---

# 52. Phase B — richer representation learning

### B1
Freeze I2-B research protocol.

### B2
Define 2D slice dataset.

### B3
Create PCA/POD baseline.

### B4
Implement simple convolutional autoencoder.

### B5
Track all runs in MLflow.

### B6
Run true multi-seed evaluation.

### B7
Evaluate held-out operating conditions.

### B8
Compare physical statistics / modal structure.

### B9
Write result to `FINDINGS.md`.

### Decision gate

Continue nonlinear representation learning **only if evidence justifies it**.

---

# 53. Phase C — virtual-sensor / time-series ML

### C1
Select physical sensors/variables.

### C2
Create standardized temporal dataset builder.

### C3
Persist dataset in industry-friendly tabular format where appropriate.

### C4
Implement leakage-safe chronological splits.

### C5
Persistence baseline.

### C6
Autoregressive / linear baseline.

### C7
XGBoost or LightGBM.

### C8
LSTM.

### C9
OOD operating-condition tests.

### C10
Physics-aware / regime-aware evaluation.

### C11
Track experiments and register validated model.

### Exit criterion

METIS contains a credible end-to-end time-series forecasting workflow.

---

# 54. Phase D — SQL/data layer

This can partly overlap Phase C.

### D1
Introduce SQLite locally.

### D2
Store structured:

- case metadata;
- sensor metadata;
- experiment metadata;
- predictions;
- metrics.

### D3
Create reproducible analytical queries.

### D4
Document schema.

### D5
Use SQL naturally in analysis/report generation.

### Exit criterion

SQL is a real part of the project rather than a résumé keyword.

---

# 55. Phase E — production-style model path

### E1
Freeze one validated forecasting model.

### E2
Implement inference package.

### E3
Implement FastAPI.

### E4
Unit/integration tests.

### E5
Dockerize inference.

### E6
CI Docker/API smoke test.

### E7
Implement logging.

### E8
Implement OOD/data-drift diagnostics.

### E9
Deploy to cloud.

### E10
Document architecture and deployment.

### Exit criterion

A remote client can send a valid input and receive a versioned model prediction from a containerized deployed service.

---

# 56. Phase F — advanced models

Only after Phases A–E demonstrate value.

Candidates:

- full-field autoencoder;
- FNO;
- DeepONet;
- U-Net + conditioning;
- WNO.

Each requires a hypothesis and baseline.

---

# 57. Recommended next 2 weeks

## Day 1–2

Fix:

- autoencoder seed initialization;
- seed regression test.

Rerun representation study.

Update:

- MLflow;
- JSON outputs;
- `FINDINGS.md`.

## Day 3

Fix:

- persistent benchmark cache.

## Day 4

Fix:

- source/cache fingerprint.

## Day 5

Fix:

- `CaseRegistry.from_config()`;
- strict validation policy.

### Gate

Run full suite.

No known P0/P1 reproducibility issue should remain.

---

## Week 2

Freeze I2-B protocol.

Define:

- selected field;
- selected slice;
- sample unit;
- train cases;
- validation samples;
- OOD cases;
- latent dimensions;
- PCA/POD baseline;
- autoencoder architecture;
- evaluation metrics;
- stop criteria.

Then build the dataset artifact.

Do **not** tune the neural model before the protocol is written.

---

# 58. Recommended 2–3 month target

A strong state after the coming months would be:

## Scientific

- regime-v1 benchmark frozen;
- case-level AE negative result corrected/finalized;
- slice-level representation study complete;
- one publication direction clearly identified.

## Group platform

- reusable case/data infrastructure;
- validated analysis commands;
- cached datasets;
- CLI;
- documentation;
- reports;
- student-ready workflows.

## Industry ML

- MLflow;
- trainable models;
- multi-seed experiments;
- temporal forecasting;
- XGBoost/LSTM;
- SQL;
- artifact/model registry;
- rigorous OOD evaluation.

## Production

- FastAPI;
- Docker;
- CI/CD;
- logging;
- drift/OOD monitoring;
- one cloud deployment.

That combination would materially change the profile from:

> **strong computational PhD with emerging ML**

to:

> **computational scientist / applied ML engineer with demonstrated end-to-end ML platform and deployment experience.**

---

# 59. Suggested CV outcome

Once the relevant milestones are genuinely complete, a defensible bullet could become:

> **Developed METIS, a reusable ML platform for multi-terabyte high-fidelity simulation data, covering validated ingestion, leakage-safe preprocessing, versioned dataset generation, experiment tracking, model training, physics-aware and OOD evaluation, automated testing/CI, artifact management, and reproducible analysis workflows.**

After deployment:

> **Productionized a multivariate temporal forecasting model through versioned inference, FastAPI, Docker, CI/CD, cloud deployment, logging, and data/model-shift monitoring.**

These should only be used after the implementation exists.

---

# 60. Scope guardrail

The main strategic risk from this point is:

> **continuing vertically into increasingly sophisticated academic models while leaving the ML lifecycle incomplete.**

Avoid:

```text
AE
→ VAE
→ FNO
→ WNO
→ DeepONet
→ paper
```

as the only progression.

Preferred progression:

```text
validated DNS
    ↓
reusable dataset
    ↓
baseline
    ↓
ML model
    ↓
experiment tracking
    ↓
physics/OOD evaluation
    ↓
temporal industrial task
    ↓
versioned model
    ↓
API
    ↓
Docker
    ↓
cloud
    ↓
monitoring
```

This horizontal progression through the ML lifecycle is now the highest-value path for both the project and the academia-to-industry transition.

---

# 61. Final priority list

| Priority | Work item | Why |
|---:|---|---|
| **1** | Fix AE seed bug + rerun | Scientific correctness |
| **2** | Fix persistent caching | Reproducibility/performance |
| **3** | Strengthen cache fingerprints | Data lineage |
| **4** | Fix config-root behavior | API correctness |
| **5** | Strict validation default | Group safety |
| **6** | I2-B slice representation study | Next scientific ML contribution |
| **7** | Virtual-sensor dataset | Industry bridge |
| **8** | Forecasting baselines + XGBoost/LSTM | Conventional applied ML |
| **9** | SQL layer | Common industry gap |
| **10** | Register validated temporal model | Model lifecycle |
| **11** | FastAPI inference | Production interface |
| **12** | Docker | Reproducible deployment |
| **13** | CI/CD deployment path | Engineering maturity |
| **14** | OOD/drift monitoring | Scientific + industry value |
| **15** | Cloud deployment | Close cloud gap |
| **16** | FNO/DeepONet/etc. | Only when scientifically justified |

---

# 62. Immediate next action

The next commit should be about **correctness**, not new capability.

Recommended commit sequence:

```text
fix: seed autoencoder initialization from experiment seed
test: verify independent model initializations across seeds
research: rerun case-level representation benchmark
docs: update representation findings
```

Then:

```text
fix: persist regime-v1 feature artifacts
feat: strengthen dataset source fingerprinting
fix: honor mapping data.root in CaseRegistry
feat: abort analyses on validation errors by default
```

After those commits:

> **freeze infrastructure and begin I2-B.**

That is the point where new ML research should again become the dominant work.

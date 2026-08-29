# Roadmap — Reusable ML Surrogate Solver for High-Pressure Transcritical Flow Data

## 1. Project vision

Develop a reusable **data-driven surrogate solver framework** for high-pressure transcritical flow problems that allows researchers and students to train, evaluate, deploy, and reuse machine-learning models on the existing DNS database without rebuilding the complete workflow for each new study.

The objective is not only to train a specific neural network. The project should create a **research infrastructure** that behaves like a new numerical solver or modelling platform:

> **Input physical conditions and flow history → select/train a validated surrogate model → predict quantities of interest → quantify confidence and physical consistency → compare against reference DNS.**

Once the framework is established, future Bachelor, Master, PhD, or visiting students should be able to use it to run new studies, benchmark architectures, test operating conditions, or investigate new physical questions without recreating the data and ML infrastructure.

---

# 2. Internal pitch to the research group

## Core argument

The group already owns a validated, high-value DNS database covering multiple thermodynamic operating conditions. At present, exploiting this database for machine learning requires substantial manual work:

- loading and restructuring large simulation datasets;
- defining train/validation/test partitions;
- preprocessing high-dimensional fields;
- implementing models independently;
- reproducing benchmarks;
- comparing different architectures;
- validating physical consistency;
- handling out-of-distribution cases;
- managing experiments and model versions.

This project converts those one-off tasks into a **reusable scientific platform**.

The expected result is analogous to developing a new solver:

- the underlying DNS remains the high-fidelity reference;
- the surrogate framework becomes a lower-cost predictive layer;
- new users can define operating conditions and modelling tasks;
- models can be trained or loaded through a standardized interface;
- predictions can be benchmarked against reference solutions;
- new architectures can be incorporated as modules.

## Why the group should invest time in it

1. **Reusable infrastructure**  
   Future students can run ML studies without rebuilding the pipeline.

2. **Faster exploitation of existing DNS investment**  
   The current database becomes substantially easier to reuse for new publications and student projects.

3. **Lower entry barrier for MSc/BSc projects**  
   Students can focus on the research question rather than spending months building data infrastructure.

4. **Consistent benchmarking**  
   Different architectures can be compared using the same preprocessing, data splits, metrics, and physical constraints.

5. **Publication potential**  
   The first scientific study can focus on robustness and generalization of surrogate models across transcritical regimes.

6. **Foundation for future funding**  
   The platform can support future proposals involving reduced-order modelling, Scientific ML, digital twins, optimization, or real-time prediction.

---

# 3. Scientific framing

## Proposed central research question

> **How reliably can data-driven surrogate models predict high-pressure transcritical flow behaviour across changing thermodynamic operating conditions, and how can we detect when a model is operating outside the regime in which its predictions can be trusted?**

This creates a stronger contribution than simply reporting prediction accuracy for one neural architecture.

The project should focus on:

- generalization across operating conditions;
- interpolation versus extrapolation;
- distribution shift;
- physical consistency;
- model robustness;
- uncertainty and trustworthiness;
- computational efficiency relative to DNS;
- reproducible architecture benchmarking.

---

# 4. Main project outputs

The project should intentionally produce **two parallel outputs**.

## 4.1 Scientific output

A publication centred on:

**Robust surrogate modelling under thermodynamic regime shift in high-pressure transcritical turbulence**

Possible scientific contributions:

- systematic comparison of surrogate architectures;
- interpolation versus extrapolation across thermodynamic conditions;
- OOD evaluation;
- physics-consistency metrics;
- relationship between physical regime change and ML model degradation;
- modal-space or latent-space diagnostics for detecting unreliable predictions;
- comparison between full-field and sensor-level prediction;
- inference-cost versus accuracy trade-offs.

## 4.2 Engineering output

A reusable open or group-internal framework that includes:

- standardized data ingestion;
- dataset generation;
- preprocessing;
- experiment configuration;
- baseline models;
- neural-network architectures;
- training;
- evaluation;
- model registry/versioning;
- inference;
- testing;
- monitoring;
- documentation;
- reproducible examples.

The engineering framework should remain useful even after the first paper is finished.

---

# 5. Conceptual architecture

```text
                    ┌─────────────────────────┐
                    │       DNS DATABASE      │
                    │  validated reference    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     DATA INGESTION      │
                    │ HDF5 / metadata / QA    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DATASET GENERATION    │
                    │ fields / sensors / QoI  │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌─────────────────────┐       ┌─────────────────────┐
       │  FULL-FIELD MODELS  │       │ SENSOR/TIME-SERIES  │
       │ FNO / WNO / U-Net   │       │ LSTM / XGBoost /    │
       │ DeepONet / baselines│       │ statistical models  │
       └──────────┬──────────┘       └──────────┬──────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ TRAINING + EXPERIMENTS  │
                    │ MLflow / configs / logs │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       EVALUATION        │
                    │ accuracy / OOD / physics│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      MODEL REGISTRY     │
                    │ validated model versions│
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ BATCH INFERENCE  │      │ API / USER TOOL  │
          └────────┬─────────┘      └────────┬─────────┘
                   │                         │
                   └────────────┬────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ MONITORING / OOD CHECK  │
                    │ drift / physics / trust │
                    └─────────────────────────┘
```

---

# 6. User workflow

The long-term goal should be that another student can interact with the framework approximately as follows:

```bash
# Create a dataset
python -m solver.data.build_dataset \
    --cases A B C \
    --variables u T rho cp \
    --task forecasting

# Train a model
python -m solver.train \
    --model fno \
    --config configs/fno_caseABC.yaml

# Evaluate interpolation and OOD behaviour
python -m solver.evaluate \
    --run-id <model_id> \
    --cases D E \
    --ood

# Run predictions
python -m solver.predict \
    --model <validated_model> \
    --input new_case.h5

# Start inference service
docker compose up
```

The exact interface can evolve, but the project should preserve this philosophy:

> **A new user should configure a study, not rewrite the platform.**

---

# 7. Research tasks

## Track A — Full-field surrogate solver

### Objective

Predict the evolution or reconstruction of high-dimensional flow fields.

Possible formulation:

\[
q(x,y,z,t-n:t)
\rightarrow
q(x,y,z,t+\Delta t)
\]

Variables may include:

- velocity components;
- temperature;
- density;
- thermodynamic properties;
- derived quantities where scientifically justified.

### Candidate models

Baseline first:

- persistence;
- linear reduced-order model;
- POD-based reconstruction/prediction.

ML:

- LSTM-based reduced-order model;
- FNO;
- WNO;
- U-Net + conditioning / FiLM;
- DeepONet.

### Questions

- Which architectures interpolate best?
- Which extrapolate best?
- How does error vary with prediction horizon?
- How does performance change with pressure and wall-temperature conditions?
- Do models preserve physically important flow statistics?
- How much acceleration is achieved relative to DNS?

---

# 8. Research Track B — Virtual sensor forecasting

## Objective

Transform the DNS database into a sensor-like industrial forecasting problem.

Virtual probes can be placed at physically meaningful locations, for example:

- cold-wall viscous/buffer region;
- cold-wall outer region;
- channel centre;
- hot-wall outer region;
- hot-wall buffer/viscous region.

Possible signals:

- \(u(t)\);
- \(v(t)\);
- \(w(t)\);
- \(T(t)\);
- \(\rho(t)\);
- \(c_p(t)\);
- wall quantities;
- selected derived observables.

Example task:

\[
X_{t-L:t}
\rightarrow
X_{t+1:t+H}
\]

where:

- \(L\) = historical context window;
- \(H\) = forecast horizon.

## Candidate models

### Baselines

- persistence;
- moving average;
- autoregressive model;
- linear regression with lag features.

### Classical ML

- Random Forest;
- XGBoost / LightGBM;
- gradient boosting with lagged/statistical features.

### Deep learning

- LSTM;
- temporal convolution;
- selected transformer architecture if scientifically justified.

## Why this track matters

This gives the project a direct connection to:

- sensor analytics;
- forecasting;
- predictive maintenance;
- industrial monitoring;
- anomaly detection;
- digital twins.

It also creates accessible MSc/BSc projects requiring much less computational cost than full-field neural operators.

---

# 9. OOD and domain-shift experiments

This should be one of the defining components of the project.

## 9.1 Random split

Use only as a basic reference.

It should **not** be the main evidence of model generalization.

## 9.2 Temporal holdout

Train on earlier temporal windows and evaluate on unseen future windows.

Purpose:

- detect temporal leakage;
- test temporal robustness.

## 9.3 Operating-condition interpolation

Train on multiple operating conditions and evaluate on held-out cases inside the sampled envelope.

## 9.4 Pressure extrapolation

Example:

```text
Train: low/intermediate pressure
Test: highest-pressure regime
```

and vice versa where scientifically meaningful.

## 9.5 Thermal-condition extrapolation

Train on moderate wall-temperature differences and test on stronger thermal forcing.

## 9.6 Thermodynamic-regime shift

Train where pseudo-boiling effects are stronger and test where their influence changes significantly.

The exact case combinations should be defined from the physical interpretation of the DNS database rather than from arbitrary ML splitting.

---

# 10. Physical validation

Pure ML metrics are insufficient for the publication.

## Standard prediction metrics

- MAE;
- RMSE;
- normalized RMSE;
- \(R^2\);
- relative \(L_2\) error.

## Physical/statistical metrics

Depending on the predicted variables:

- mean profiles;
- RMS profiles;
- Reynolds stresses;
- TKE;
- temperature fluctuations;
- wall shear;
- energy spectra;
- dominant wavelengths;
- dominant frequencies.

## Modal comparison

Leverage the existing POD/SPOD expertise.

Possible measures:

- modal energy recovery;
- mode similarity;
- principal angles;
- subspace distance;
- spectral-content preservation.

---

# 11. OOD detection and model trustworthiness

A model should not only produce a prediction. It should indicate when the prediction is likely to be unreliable.

Candidate diagnostics:

## Input-space drift

- statistical distance between training and inference distributions;
- feature-level drift;
- thermodynamic-condition distance.

## Latent-space drift

Measure the distance between new samples and the training manifold.

## Model disagreement

Use ensembles or multiple architectures.

## Physical residuals

Evaluate whether predictions violate expected physical constraints.

## POD/SPOD subspace distance

Potential novel direction:

> Test whether divergence between the modal subspace of incoming data and the training-data subspace predicts surrogate-model error.

Example concept:

\[
D_{\mathrm{subspace}}
\left(
\Phi_{\mathrm{train}},
\Phi_{\mathrm{new}}
\right)
\rightarrow
\text{expected prediction reliability}
\]

This could become a particularly distinctive scientific contribution.

---

# 12. Engineering architecture

Recommended repository structure:

```text
transcritical-surrogate-solver/
│
├── README.md
├── pyproject.toml
├── docker-compose.yml
│
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── src/
│   └── surrogate_solver/
│       │
│       ├── data/
│       │   ├── ingestion/
│       │   ├── validation/
│       │   ├── preprocessing/
│       │   └── datasets/
│       │
│       ├── features/
│       │
│       ├── models/
│       │   ├── baselines/
│       │   ├── classical_ml/
│       │   ├── lstm/
│       │   ├── fno/
│       │   ├── wno/
│       │   └── deeponet/
│       │
│       ├── training/
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── physics.py
│       │   ├── ood.py
│       │   └── modal.py
│       │
│       ├── registry/
│       │
│       ├── inference/
│       │
│       ├── monitoring/
│       │
│       └── api/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── scripts/
│
├── notebooks/
│   └── exploratory_only/
│
├── examples/
│   ├── sensor_forecasting/
│   └── full_field_prediction/
│
└── .github/
    └── workflows/
```

---

# 13. Design principles

## 13.1 Configuration over code duplication

A new experiment should be defined primarily through configuration files.

Example:

```yaml
dataset:
  train_cases: [A, B, C]
  validation_cases: [D]
  test_cases: [E]
  variables: [u, T, rho]
  history: 100
  horizon: 20

model:
  type: lstm
  hidden_size: 128
  layers: 3

training:
  epochs: 100
  batch_size: 64

evaluation:
  physics_metrics: true
  ood_metrics: true
```

## 13.2 Models should share a common interface

Conceptually:

```python
model.fit(train_data)
prediction = model.predict(data)
model.save(path)
model.load(path)
```

This makes benchmark studies substantially easier.

## 13.3 Notebooks are for exploration

Production research workflows should live in reusable Python modules.

Notebooks should mainly be used for:

- initial exploration;
- figures;
- demonstration examples.

## 13.4 Every result should be reproducible

Every experiment should record:

- dataset version;
- model configuration;
- Git commit;
- random seed;
- metrics;
- hardware;
- training time;
- inference time.

---

# 14. MLOps layer

## Experiment tracking

Use MLflow to store:

- hyperparameters;
- model versions;
- metrics;
- artifacts;
- plots;
- training metadata.

## Model registry

A model should only become a validated solver model after passing defined acceptance tests.

Possible lifecycle:

```text
Development
   ↓
Validated
   ↓
Candidate
   ↓
Production / Approved
   ↓
Archived
```

"Production" in the research-group context means:

> validated and approved for reuse by other researchers.

It does not need to imply a commercial live system.

---

# 15. Testing strategy

## Unit tests

Examples:

- HDF5 reader returns correct shapes;
- normalization is invertible;
- sensor extraction selects correct coordinates;
- model output dimensions are correct;
- metric calculations match reference values.

## Integration tests

Examples:

```text
raw sample
→ preprocessing
→ tiny training run
→ prediction
→ evaluation
```

## Regression tests

Store small reference datasets and ensure that future code changes do not unexpectedly modify validated outputs.

This is particularly valuable for a solver-like research platform.

---

# 16. CI/CD

Use GitHub Actions initially.

Pipeline:

```text
commit / pull request
        ↓
lint
        ↓
unit tests
        ↓
integration test
        ↓
build package
        ↓
build Docker image
```

Later:

```text
validated release
        ↓
model/API deployment
```

---

# 17. Inference service

A simple FastAPI layer can expose validated surrogate models.

Possible API:

```text
GET /health

GET /models

POST /predict

POST /predict/sensors

POST /ood-check
```

Example request:

```json
{
  "model": "fno_v3",
  "case_conditions": {
    "pressure_ratio": 2.0,
    "cold_wall_temperature_ratio": 0.8,
    "hot_wall_temperature_ratio": 1.4
  },
  "input_data": "..."
}
```

For the research group, batch inference will probably remain more important than real-time API inference.

The API exists primarily to:

- standardize access;
- demonstrate portability;
- enable future tools or interfaces;
- provide production-style engineering experience.

---

# 18. Containerization

Docker should package:

- dependencies;
- inference code;
- model interface;
- API.

A future student should ideally be able to run:

```bash
git clone <repository>
docker compose up
```

and access a working example.

GPU and HPC training can remain outside Docker initially if cluster constraints make containerization inconvenient.

---

# 19. Cloud deployment

This should be a later phase, not the starting point.

Potential target:

- Azure;
- AWS;
- GCP.

Given recurring industry demand for Azure/Databricks, Azure is a reasonable candidate.

Minimum objective:

- deploy inference API;
- host one validated model;
- configure logging;
- demonstrate remote inference;
- document deployment.

The full DNS database does not need to move to the cloud.

Use a reduced demonstration dataset.

---

# 20. Monitoring

Monitoring should have two interpretations.

## Engineering monitoring

- inference latency;
- failures;
- API health;
- model version;
- logging.

## Scientific monitoring

- input-distribution drift;
- OOD score;
- prediction confidence;
- physical constraint violations;
- expected model reliability.

The second is the scientifically interesting component.

---

# 21. Student-project architecture

The framework should intentionally support future student work.

## Example Bachelor projects

### BSc-1 — Baseline forecasting

Benchmark classical methods for virtual-sensor forecasting.

### BSc-2 — Data drift detection

Implement and compare statistical drift metrics.

### BSc-3 — API / visualization interface

Develop a user interface for running validated models.

### BSc-4 — Model performance benchmark

Compare inference speed on CPU/GPU.

---

## Example Master projects

### MSc-1 — Neural operator benchmarking

Compare FNO, WNO, DeepONet, and U-Net variants.

### MSc-2 — OOD generalization

Study interpolation and extrapolation across thermodynamic conditions.

### MSc-3 — Physics-aware loss functions

Introduce conservation or thermodynamic consistency constraints.

### MSc-4 — Uncertainty quantification

Implement ensemble or probabilistic uncertainty methods.

### MSc-5 — Reduced-order digital twin

Combine POD/SPOD representations with temporal ML models.

---

# 22. Development roadmap

## Phase 0 — Scientific definition

**Duration target: 1–2 weeks**

### Tasks

- define primary prediction task;
- define scientific hypotheses;
- choose cases;
- define interpolation/OOD splits;
- define target variables;
- define baseline methods;
- determine computational budget;
- define paper-level evaluation metrics.

### Deliverable

`research_protocol.md`

No serious model development should begin before this exists.

---

# 23. Phase 1 — Repository and data layer

**Duration target: 2–4 weeks**

### Build

- Python package structure;
- configuration system;
- HDF5 readers;
- metadata schema;
- dataset validation;
- train/validation/test split manager;
- normalization;
- sensor extraction;
- reproducible dataset generation.

### Tests

- data integrity;
- shape consistency;
- deterministic dataset generation;
- normalization.

### Milestone

A new user can generate a dataset from selected DNS cases with one command.

---

# 24. Phase 2 — Baseline solver

**Duration target: 2–3 weeks**

Implement simple models first.

### Sensor track

- persistence;
- autoregression;
- linear regression;
- XGBoost/LightGBM.

### Full-field track

- persistence;
- POD reconstruction;
- simple reduced-order temporal model.

### Milestone

The framework can train, evaluate, and compare baseline models end-to-end.

This creates the minimum viable solver.

---

# 25. Phase 3 — Experiment infrastructure

**Duration target: 1–2 weeks**

Introduce:

- MLflow;
- experiment configuration;
- run IDs;
- model artifacts;
- metric storage;
- reproducibility metadata.

### Milestone

Every experiment can be reproduced from a configuration file and run identifier.

---

# 26. Phase 4 — Advanced ML models

**Duration target: 4–8 weeks**

Add models incrementally.

Recommended order:

1. LSTM / temporal neural baseline;
2. FNO;
3. U-Net + FiLM;
4. WNO;
5. DeepONet.

Do not implement all architectures simultaneously.

Each model must use the common solver interface.

### Milestone

At least two strong ML models are benchmarked against simple baselines using identical data and evaluation.

---

# 27. Phase 5 — OOD generalization study

**Duration target: 3–5 weeks**

Run the scientific experiment matrix.

Evaluate:

- in-distribution performance;
- interpolation;
- pressure extrapolation;
- thermal extrapolation;
- regime shift;
- error versus forecast horizon;
- error versus physical operating point.

### Milestone

First paper-quality result set.

---

# 28. Phase 6 — Physics-aware evaluation

**Duration target: 3–4 weeks**

Implement:

- physical statistics;
- spectra;
- POD/SPOD comparisons;
- conservation/consistency metrics where applicable.

### Milestone

Determine whether low pointwise error actually corresponds to physically correct predictions.

---

# 29. Phase 7 — Model trust / OOD detection

**Duration target: 3–5 weeks**

Implement selected approaches:

- feature drift;
- latent-space distance;
- ensemble disagreement;
- modal-subspace distance;
- physics-residual indicators.

Research question:

> Can model degradation be detected without access to the reference DNS solution?

### Milestone

Candidate methodological novelty for publication.

---

# 30. Phase 8 — Solver packaging

**Duration target: 2–4 weeks**

Add:

- CLI;
- model registry;
- standardized prediction interface;
- documentation;
- examples;
- release versions.

### Milestone

Another group member can use a validated model without understanding its training implementation.

---

# 31. Phase 9 — Software reliability

**Duration target: 2–3 weeks**

Add:

- pytest suite;
- integration tests;
- regression tests;
- linting;
- GitHub Actions;
- automated package builds.

### Milestone

Code changes automatically undergo validation before merge.

---

# 32. Phase 10 — Deployment layer

**Duration target: 2–4 weeks**

Implement:

- FastAPI;
- Docker;
- Docker Compose;
- inference logging;
- `/health`;
- `/models`;
- `/predict`;
- `/ood-check`.

### Milestone

A validated surrogate model can be executed as a standalone service.

---

# 33. Phase 11 — Cloud demonstration

**Duration target: 1–3 weeks**

Deploy a reduced demonstration model to one cloud provider.

Objectives:

- remote inference;
- model artifact storage;
- environment configuration;
- logs;
- documented deployment procedure.

This is not essential for the first publication.

It is important for the project's engineering maturity.

---

# 34. Phase 12 — Publication and handover

Prepare:

- paper;
- reproducibility package;
- documentation;
- tutorial;
- example datasets;
- student onboarding guide.

A new student should be given:

1. repository;
2. documentation;
3. example case;
4. exercise.

Target:

> They should successfully train or run a model during their first days rather than their first months.

---

# 35. Suggested publication structure

## Working title

**Robust Data-Driven Surrogate Modelling of High-Pressure Transcritical Turbulence Across Thermodynamic Regime Shifts**

Alternative:

**Towards Trustworthy Neural Surrogate Solvers for High-Pressure Transcritical Flows**

## Possible paper structure

1. Introduction
2. DNS database
3. Reusable surrogate framework
4. Prediction tasks
5. Baseline and ML architectures
6. Validation methodology
7. In-distribution results
8. Cross-condition generalization
9. Physics-consistency analysis
10. OOD detection / model trustworthiness
11. Computational cost
12. Conclusions

The paper should focus on the scientific results.

The repository should demonstrate the broader engineering architecture.

---

# 36. Minimum viable publishable project

Avoid making the first paper dependent on completing the entire platform.

The minimum publishable scope should be:

- standardized dataset pipeline;
- at least one baseline;
- at least two ML models;
- systematic operating-condition splits;
- OOD evaluation;
- physics-aware validation;
- computational-cost comparison.

Everything else can progressively strengthen the platform.

---

# 37. Minimum viable engineering project

For industry relevance, the minimum engineering layer should contain:

- installable Python package;
- reproducible configuration;
- proper train/evaluate commands;
- tests;
- Git;
- GitHub Actions;
- MLflow;
- FastAPI;
- Docker;
- model registry/versioning;
- basic monitoring.

This is enough to make a substantial difference to the professional profile.

---

# 38. Scope control

## Do now

- Python package architecture;
- clean data ingestion;
- reproducible splits;
- baselines;
- MLflow;
- PyTorch models;
- OOD methodology;
- physics validation;
- tests;
- GitHub Actions.

## Do after the scientific core works

- FastAPI;
- Docker;
- model registry;
- drift monitoring;
- cloud deployment.

## Do only if justified

- Kubernetes;
- Kafka;
- Spark;
- Databricks;
- Airflow;
- Terraform;
- complex frontend.

These should not delay the research.

---

# 39. Success criteria

## Scientific success

The project demonstrates:

- statistically and physically rigorous model comparison;
- clear differences between interpolation and extrapolation;
- meaningful analysis of generalization failure;
- at least one useful model-reliability indicator;
- publication-quality findings.

## Group-level success

The project provides:

- reusable research infrastructure;
- lower barrier for future student projects;
- consistent model benchmarking;
- documented workflows;
- extensible architecture.

## Engineering success

The repository demonstrates:

- modular Python;
- testing;
- CI/CD;
- experiment tracking;
- model versioning;
- API inference;
- Docker;
- monitoring;
- optional cloud deployment.

## Personal career success

After completion, it should be accurate to state:

> Developed a reusable end-to-end ML surrogate platform for high-dimensional physical data, covering data ingestion, model training, experiment tracking, OOD evaluation, physics-aware validation, model versioning, automated testing, CI/CD, containerized inference, and monitoring.

---

# 40. Recommended first decision

Before coding the full architecture, decide which task will become the **reference example** for the framework.

Recommended:

> **Virtual-sensor multivariate forecasting under thermodynamic domain shift**

Why:

- computationally cheaper than full-field training;
- fast iteration;
- easy to benchmark;
- ideal for testing the architecture;
- directly relevant to industry;
- accessible to future students;
- naturally supports OOD/drift research.

Once the complete architecture works on this task, integrate full-field neural operators as the high-dimensional solver track.

This produces a progression similar to conventional solver development:

```text
simple canonical problem
        ↓
validation
        ↓
software architecture
        ↓
advanced physical problem
        ↓
general reusable solver
```

---

# 41. Immediate next steps

## Week 1

- write the research hypothesis;
- choose the reference forecasting problem;
- select sensor locations and variables;
- define temporal windows;
- define train/validation/OOD cases;
- define baseline metrics.

## Week 2

- create repository architecture;
- implement DNS metadata representation;
- implement one HDF5 reader;
- implement dataset generation;
- write first unit tests.

## Week 3

- persistence baseline;
- autoregressive baseline;
- first XGBoost/LightGBM model;
- reproducible evaluation command.

## Week 4

- add MLflow;
- add experiment configuration;
- generate first full benchmark table.

At that point, there should already be enough evidence to decide whether the project is scientifically promising before investing in the more expensive neural architectures.

---

# 42. One-sentence pitch

> **Develop a reusable machine-learning surrogate solver for the group's validated transcritical DNS database, so future researchers and students can configure, train, validate, and deploy predictive models across operating conditions through a common framework rather than rebuilding the ML pipeline for every project.**

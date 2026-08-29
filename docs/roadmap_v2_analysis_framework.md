# Roadmap — Reusable ML Analysis & Physical-Discovery Framework for High-Pressure Transcritical DNS

## 1. Project vision

Develop a reusable **machine-learning analysis and physical-discovery framework** that sits downstream of the research group's existing high-fidelity DNS solver.

The group already has the capability to generate high-pressure transcritical flow data through DNS. This project does **not** aim to replace that solver. Instead, it creates the complementary data-driven layer required to systematically exploit the raw DNS output.

The intended workflow is:

> **Physical configuration → DNS solver → raw DNS fields → ML analysis engine → physically meaningful structures, trends, reduced representations, predictive relations, regime changes, and new hypotheses.**

The framework should ingest raw or minimally processed DNS output and provide standardized modules for:

- data ingestion and quality control;
- preprocessing and derived-variable generation;
- first- and second-order statistics;
- spectral and time-frequency analysis;
- POD/SPOD and other reduced representations;
- cross-variable and cross-case comparison;
- machine-learning-based feature extraction and regime discovery;
- temporal prediction and forecasting where scientifically useful;
- out-of-distribution and domain-shift analysis;
- physics-aware model validation;
- automated reproducible reporting.

The objective is not to build one neural network for one paper. The objective is to create a **reusable research analysis engine** that future Bachelor, Master, PhD, or visiting students can use to investigate new DNS cases without reconstructing the entire data-processing and ML workflow from scratch.

The long-term vision is analogous to creating a new solver ecosystem, but with a different role:

- the **DNS solver** generates high-fidelity physics;
- the **ML framework** mines, organizes, compares, models, and interprets the generated physics.

A new user should primarily configure a study and choose analysis modules rather than rewrite low-level infrastructure.

---

# 2. Internal pitch to the research group

## Core argument

The group already owns two major assets:

1. a validated DNS solver capable of producing high-pressure transcritical flow data;
2. a growing database of expensive, high-fidelity simulations across multiple thermodynamic operating conditions.

The bottleneck is increasingly **not data generation alone**, but efficient and reproducible extraction of physical knowledge from those datasets.

At present, each new analysis or student project can require substantial repeated work:

- understanding native solver output;
- loading and restructuring large HDF5 datasets;
- reconstructing metadata and operating conditions;
- computing derived quantities;
- implementing statistics and spectral tools independently;
- repeating POD/SPOD or other reduced-order analyses;
- writing case-comparison scripts;
- defining ML datasets and train/validation/test partitions;
- benchmarking models inconsistently;
- reproducing figures and metrics;
- validating physical consistency;
- tracking experiments manually.

This project converts those one-off workflows into a **shared scientific platform**.

## The “new solver” analogy

The project can be sold internally as the development of a new reusable computational capability:

```text
Existing capability
-------------------
Physical problem
      ↓
DNS solver
      ↓
Raw high-fidelity fields


New capability
--------------
Raw high-fidelity fields
      ↓
ML / data-driven analysis engine
      ↓
Physics extraction
      ↓
Cross-case comparison
      ↓
ML-based discovery / prediction
      ↓
Physical insight and new hypotheses
```

The ML framework therefore behaves like a **post-processing and discovery solver**:

- standardized inputs;
- modular algorithms;
- reproducible configuration;
- automated execution;
- validated outputs;
- extensible architecture;
- documented interfaces for future users.

## Why the group should invest time in it

1. **Reusable infrastructure**  
   Future students can analyse new DNS campaigns without rebuilding the data pipeline.

2. **Higher return on existing DNS investment**  
   Expensive simulations become easier to reuse for multiple papers, theses, and student projects.

3. **Lower entry barrier for BSc/MSc projects**  
   Students can focus on a scientific question rather than spending months learning file formats and rewriting analysis scripts.

4. **Standardized physics extraction**  
   Statistics, spectra, POD/SPOD, feature extraction, and cross-case comparisons can use common validated implementations.

5. **Consistent ML benchmarking**  
   Different ML methods can use identical preprocessing, splits, metrics, physical checks, and experiment tracking.

6. **Foundation for automated scientific discovery**  
   The platform can evolve from conventional post-processing toward clustering, representation learning, regime identification, anomaly detection, and AI-assisted hypothesis generation.

7. **Publication potential**  
   The first paper can focus on a narrow scientific question while the software remains a broader group asset.

8. **Foundation for future funding**  
   The platform can support proposals involving Scientific ML, reduced-order modelling, digital twins, optimization, uncertainty quantification, or intelligent simulation workflows.

## What the framework is not

The first objective is **not**:

- replacing the DNS solver;
- claiming ML can reproduce all DNS physics;
- building a commercial real-time product;
- forcing neural networks into analyses where established physical methods are superior.

Surrogate prediction can be one module inside the platform, but the primary purpose is:

> **to transform raw DNS output into reproducible, interpretable, and extensible physical knowledge.**

---

# 3. Scientific framing

The platform itself is broad, but each publication built on top of it should answer a **narrow and falsifiable physical or methodological question**.

## Primary scientific direction

A strong first direction is:

> **Can data-driven representations identify and quantify physically meaningful regime changes across high-pressure transcritical DNS cases, and can those representations reveal when apparently similar flows belong to different thermodynamic or dynamical regimes?**

This naturally connects the existing expertise in:

- high-pressure transcritical physics;
- POD/SPOD;
- spectral organization;
- thermophysical-property variation;
- pseudo-boiling effects;
- cross-variable coupling;
- high-fidelity DNS;
- neural and reduced-order representations.

## Candidate research questions

The framework should make questions such as these straightforward to study:

### Physical-regime identification

- Can unsupervised methods recover known thermodynamic/flow regimes without explicit labels?
- Which physical variables drive the separation between regimes?
- Does the learned representation recover trends already observed through POD/SPOD?
- Are there transition regions that conventional case-by-case analysis obscures?

### Cross-condition representation learning

- Can a common latent representation organize all DNS cases?
- Which structures remain invariant across pressure and wall-temperature changes?
- Which structures are condition-specific?
- Can learned representations separate hydrodynamic and thermodynamic effects?

### Automated feature discovery

- Which temporal, spectral, thermodynamic, or modal features best distinguish cases?
- Do ML-selected features correspond to physically interpretable mechanisms?
- Can feature importance reveal overlooked coupling between variables?

### Domain shift and model trust

- How does a model trained on one subset of thermodynamic conditions behave on unseen regimes?
- Can model failure be predicted from input-space, latent-space, or modal-space distance?
- Can physical consistency metrics identify unreliable predictions?

### Temporal prediction as an optional module

Where useful, the platform can also formulate forecasting or field-prediction problems:

\[
X_{t-L:t} \rightarrow X_{t+1:t+H}
\]

or

\[
q(x,y,z,t-n:t) \rightarrow q(x,y,z,t+\Delta t)
\]

These tasks are valuable, but they should support the broader physical-analysis objective rather than define the whole project.

---

# 4. Main project outputs

The project should intentionally produce **three parallel outputs**.

## 4.1 Scientific output

One or more publications based on focused hypotheses enabled by the framework.

The first paper should preferably centre on one of:

- data-driven regime discovery across transcritical operating conditions;
- physically interpretable representation learning;
- automated cross-case feature discovery;
- OOD/model-trust diagnostics;
- comparison between conventional modal analysis and learned representations.

## 4.2 Group infrastructure output

A reusable framework that includes:

- native DNS ingestion;
- metadata handling;
- preprocessing;
- derived-variable generation;
- statistics;
- spectra;
- POD/SPOD;
- feature extraction;
- ML datasets;
- model training;
- evaluation;
- cross-case comparison;
- OOD analysis;
- automated reporting;
- documentation and examples.

The framework should remain useful even if no neural network is used in a particular study.

## 4.3 Engineering output

A maintainable software project with:

- installable Python package;
- configuration-driven workflows;
- automated tests;
- CI/CD;
- experiment tracking;
- model/version registry where applicable;
- reproducible command-line execution;
- optional API/container deployment for selected analysis services.

This engineering layer supports both scientific reproducibility and professional-grade software practice.

---

# 5. Conceptual architecture

```text
                   ┌─────────────────────────────┐
                   │   PHYSICAL CONFIGURATION    │
                   │ P, Tcw, Thw, Re, geometry   │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │        DNS SOLVER           │
                   │ Existing group capability   │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │        RAW DNS DATA         │
                   │ fields / metadata / probes  │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │    INGESTION + DATA QA      │
                   │ HDF5 / metadata / indexing  │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  STANDARD PHYSICS LAYER     │
                   │ stats / spectra / POD/SPOD  │
                   │ derived vars / correlations │
                   └──────────────┬──────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
  ┌───────────────────┐ ┌───────────────────┐ ┌────────────────────┐
  │ ML DISCOVERY      │ │ TEMPORAL MODELS   │ │ FIELD MODELS       │
  │ clustering        │ │ forecasting       │ │ neural operators   │
  │ representation    │ │ sequence models   │ │ surrogate modules  │
  │ regime detection  │ │ anomaly detection │ │ reconstruction     │
  └─────────┬─────────┘ └─────────┬─────────┘ └──────────┬─────────┘
            │                     │                      │
            └─────────────────────┼──────────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ PHYSICS-AWARE EVALUATION    │
                   │ OOD / modal / statistics    │
                   │ uncertainty / consistency   │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │ CROSS-CASE INTERPRETATION   │
                   │ trends / regimes / drivers  │
                   │ hypotheses / reports        │
                   └─────────────────────────────┘
```

The **DNS solver remains upstream and authoritative**.

The new framework begins where the DNS solver ends.

---

# 6. User workflow

The long-term objective is that another student can perform a new study approximately as follows.

```bash
# Register an existing raw DNS case
hpml ingest \
    --case case_07 \
    --input /dns/output/case_07 \
    --config configs/cases/case_07.yaml

# Validate and preprocess it
hpml preprocess case_07

# Compute standard physics
hpml analyze statistics case_07
hpml analyze spectra case_07
hpml analyze pod case_07
hpml analyze spod case_07

# Compare several operating conditions
hpml compare \
    --cases A B C D E \
    --metrics statistics spectra modal

# Build a machine-learning dataset
hpml dataset build \
    --cases A B C D E \
    --task regime_discovery

# Run a data-driven analysis
hpml train \
    --model autoencoder \
    --config configs/experiments/regime_latent_space.yaml

# Evaluate physical meaning and robustness
hpml evaluate \
    --run-id <run_id> \
    --physics \
    --ood

# Produce a standardized research report
hpml report \
    --run-id <run_id>
```

For a forecasting study:

```bash
hpml dataset build \
    --cases A B C D E \
    --task sensor_forecasting

hpml train \
    --model lstm \
    --config configs/experiments/sensor_forecasting.yaml
```

For a neural-operator study:

```bash
hpml dataset build \
    --cases A B C D E \
    --task field_prediction

hpml train \
    --model fno \
    --config configs/experiments/fno.yaml
```

The philosophy is:

> **A new user should select a physical question and configure an analysis, not rewrite the platform.**

---

# 7. Research and analysis tracks

## Track A — Standardized physics extraction and cross-case analysis

### Objective

Codify the analyses that currently exist as researcher-specific scripts into reusable, tested modules.

This should include, where supported by the available DNS output:

- first-order statistics;
- second-order statistics;
- Favre/Reynolds averages where appropriate;
- RMS quantities;
- wall quantities;
- thermophysical-property profiles;
- spectra;
- premultiplied spectra;
- POD;
- SPOD;
- DMD if scientifically useful;
- velocity-gradient invariants;
- topology indicators;
- cross-variable correlations;
- conditional statistics;
- wavelength/frequency peak extraction.

### Core scientific value

The framework should make multi-case questions easy:

- How does pressure modify a given structure?
- How does wall-temperature asymmetry alter spectral organization?
- Which quantities respond most strongly near pseudo-boiling conditions?
- Which structures persist across all cases?
- Which trends are hydrodynamic and which are thermodynamic?

### Milestone

A student should be able to reproduce the group's standard analyses for a new DNS case through configuration and validated commands rather than bespoke scripts.

---

## Track B — ML-based physical discovery

### Objective

Use machine-learning methods to discover structure across cases rather than merely predict raw fields.

Candidate methods:

- PCA/POD as linear baselines;
- autoencoders;
- variational autoencoders where justified;
- clustering;
- manifold learning;
- supervised regime classification;
- feature selection;
- sparse regression;
- representation learning;
- latent-space comparison;
- anomaly detection.

### Example outputs

- low-dimensional maps of operating regimes;
- cluster assignments;
- physically interpretable latent variables;
- discriminating features;
- transition indicators;
- cross-variable coupling patterns;
- case similarity matrices.

### Scientific requirement

Every learned representation must be compared against known physical quantities and conventional analysis.

A latent coordinate is not a physical insight until its relationship with the flow physics is demonstrated.

---

## Track C — Full-field learning and neural operators

### Objective

Use high-dimensional ML models where field prediction, compression, or representation learning adds scientific value.

Possible formulation:

\[
q(x,y,z,t-n:t)
\rightarrow
q(x,y,z,t+\Delta t)
\]

or learned compression/reconstruction:

\[
q(x,y,z,t)
\rightarrow z(t)
\rightarrow \hat q(x,y,z,t)
\]

Candidate models:

- POD/reduced-order baselines;
- convolutional autoencoders;
- FNO;
- WNO;
- U-Net + FiLM;
- DeepONet.

Scientific questions include:

- What physical structures are preserved or lost?
- How does latent structure compare with POD/SPOD?
- How does generalization change across thermodynamic conditions?
- Are failure modes associated with identifiable physical regime shifts?

---

# 8. Research Track D — Virtual-sensor forecasting and temporal modelling

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

## Why this optional track matters

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

> Test whether divergence between the modal subspace of incoming data and the training-data subspace predicts ML-model error.

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
hp-transcritical-ml/
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
│   └── hpml/
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

# 17. Optional analysis and inference service

A simple FastAPI layer can expose validated ML analysis/models.

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

# 30. Phase 8 — Framework packaging

**Duration target: 2–4 weeks**

Add:

- CLI;
- model registry;
- standardized prediction interface;
- documentation;
- examples;
- release versions.

### Milestone

Another group member can run validated analyses or ML models without understanding their internal implementation.

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

A validated ML analysis/model can be executed as a standalone service.

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

# 35. Suggested publication directions

The platform should support multiple papers. The first paper does not need to describe the entire software architecture.

## Direction A — Regime discovery

### Working title

**Data-Driven Identification of Thermodynamic and Dynamical Regimes in High-Pressure Transcritical Turbulence**

Possible structure:

1. Introduction
2. DNS database and operating conditions
3. Physical variables and standardized preprocessing
4. Conventional reference analyses
5. Data-driven representation / clustering methodology
6. Learned regime organization
7. Physical interpretation
8. Sensitivity and robustness
9. Cross-condition generalization
10. Conclusions

---

## Direction B — Learned versus modal representations

### Working title

**From POD/SPOD to Learned Representations: Data-Driven Structure Identification in High-Pressure Transcritical Turbulence**

Possible questions:

- What does nonlinear representation learning recover beyond POD/SPOD?
- Are dominant learned features physically interpretable?
- Which structures are common across thermodynamic regimes?
- Does the learned latent space organize pressure/temperature effects naturally?

---

## Direction C — Trustworthiness under regime shift

### Working title

**Detecting Loss of Validity in Data-Driven Models of High-Pressure Transcritical Flows**

Possible questions:

- How does prediction/representation quality degrade under unseen thermodynamic conditions?
- Can latent-space, modal-space, or physical metrics detect unreliable model behaviour?
- Can reliability be estimated without access to the reference DNS solution?

---

# 36. Minimum viable publishable project

Avoid making the first paper dependent on implementing the entire framework.

A strong minimum scope is:

- standardized ingestion for the selected DNS cases;
- reproducible preprocessing;
- a conventional physical baseline (e.g. POD/SPOD/statistics);
- one focused ML discovery method;
- cross-case evaluation;
- physical interpretation of the learned representation;
- robustness/OOD analysis;
- publication-quality visualization.

For example:

> Build a common feature/representation space across all selected DNS cases, determine whether known thermodynamic regimes emerge naturally, identify the variables and structures driving the separation, and test whether the representation generalizes to held-out operating conditions.

That is scientifically stronger than merely demonstrating that a neural network can fit DNS data.

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

> Developed a reusable end-to-end ML analysis platform for high-dimensional DNS data, covering raw-data ingestion, automated physical feature extraction, statistical and modal analysis, ML-based representation learning, experiment tracking, OOD evaluation, physics-aware validation, automated testing, CI/CD, and reproducible deployment.

---

# 40. Recommended first decision

Before building the full architecture, choose the **reference scientific analysis** that will prove the platform concept.

Recommended first reference task:

> **Cross-case regime discovery from raw DNS-derived physical features, benchmarked against known thermodynamic conditions and existing POD/SPOD knowledge.**

Why this is a good starting point:

- it directly matches the real objective of extracting physics from DNS;
- it uses the full value of the existing database;
- it is cheaper than full-field neural-operator training;
- it creates reusable ingestion, preprocessing, feature, and evaluation infrastructure;
- it naturally supports unsupervised learning and interpretable ML;
- it can lead to a focused publication;
- it is accessible to future students;
- it does not require pretending ML replaces the DNS solver.

Once the end-to-end architecture works on this task, add:

1. virtual-sensor temporal modelling;
2. neural-operator / full-field modules;
3. OOD/model-trust modules;
4. automated reporting;
5. optional service/cloud deployment.

This progression mirrors conventional scientific software development:

```text
validated canonical analysis
        ↓
reusable data architecture
        ↓
cross-case physical discovery
        ↓
advanced ML modules
        ↓
general group analysis platform
```

---

# 41. Immediate next steps

## Week 1 — Define the reference scientific problem

- select the first set of DNS cases;
- define the physical question;
- list raw variables available directly from the solver;
- list derived variables required;
- define metadata schema;
- decide which existing analyses will serve as physical baselines;
- define what would constitute a meaningful ML-discovered result.

## Week 2 — Build the DNS-to-analysis interface

- create repository architecture;
- implement native HDF5/DNS readers;
- implement case metadata representation;
- implement data validation;
- implement standardized coordinates/variable naming;
- add first unit tests.

## Week 3 — Reproduce established physics automatically

Implement at least:

- mean/RMS statistics;
- selected spectra;
- one POD/SPOD workflow;
- standardized cross-case comparison.

The goal is to prove that the new platform can reproduce trusted existing results.

## Week 4 — Add the first ML discovery layer

- construct case-level / local / temporal feature vectors;
- add a linear baseline;
- add clustering or representation learning;
- create first latent/regime visualizations;
- relate learned structure back to pressure, temperature, pseudo-boiling intensity, and known modal/spectral behaviour.

## Week 5+ — Decide the first paper direction

Based on the first results, choose whether the strongest publication is:

- regime discovery;
- learned versus modal representations;
- OOD/model trust;
- temporal forecasting;
- full-field learning.

Only then commit substantial compute to advanced architectures.

---

# 42. One-sentence pitch

> **Develop a reusable machine-learning analysis framework that connects directly to the group's existing high-pressure transcritical DNS solver, ingests raw simulation output, reproduces validated physical analyses, and adds data-driven tools for automated feature extraction, regime discovery, cross-case comparison, temporal modelling, and physically interpretable insight generation.**

## Short internal version

> **The DNS solver generates the physics; this framework makes the resulting physics systematically exploitable by the whole group.**

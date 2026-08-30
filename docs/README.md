# METIS documentation

## Start here

- **[getting_started.md](getting_started.md)** — install, generate mock
  data, run the tests, run your first analysis, and where to add code.
- **[architecture.md](architecture.md)** — the layers, the module map,
  and how data flows from a RHEA snapshot to a report.
- **[data_layout.md](data_layout.md)** — the DNS data directory layout,
  the `metadata.json` schema, the `[z, y, x]` axis convention, and the
  case registry.

## Extending METIS

- **[adding_an_analysis.md](adding_an_analysis.md)** — add a new
  `metis analyze <kind>` runner over the standard result shape.
- **[adding_a_model.md](adding_a_model.md)** — add a
  `RepresentationModel` and (if it trains) wire it to `Trainer`.
- **[reproducibility.md](reproducibility.md)** — config resolution,
  dataset fingerprinting/caching, MLflow tracking, the frozen benchmark,
  and provenance on every artifact.

## Research protocols

- **[i2b_representation_protocol.md](i2b_representation_protocol.md)** —
  the frozen I2-B slice-level representation-learning study (question,
  data, splits, models, evaluation, accept/stop criteria).
- Repo-root **`research_protocol.md`** — the Phase-0 regime-discovery
  reference task.

## The router-agent module (Track C/D)

- **[router_agent.md](router_agent.md)** — the standalone write-up:
  problem, the validated OOD finding it encodes, the single-tool agent
  design, and the link to Pub 4 / Pub 5.
- **[agent_implementation_plan.md](agent_implementation_plan.md)** — its
  milestone history (M0–M5).

## Background (the roadmaps this repo implements)

- **[roadmap_v1_surrogate_solver.md](roadmap_v1_surrogate_solver.md)**
- **[roadmap_v2_analysis_framework.md](roadmap_v2_analysis_framework.md)**

Research results live in the repo-root **`FINDINGS.md`**; the frozen
Phase-0 study definition in **`research_protocol.md`**; the running
implementation backlog in **`METIS_detailed_next_steps.md`**.

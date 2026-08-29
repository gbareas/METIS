# METIS — High-Pressure Transcritical ML Analysis & Discovery Framework

[![CI](https://github.com/gbareas/METIS/actions/workflows/ci.yml/badge.svg)](https://github.com/gbareas/METIS/actions/workflows/ci.yml)

**M**achine-learning **E**ngine for **T**ranscritical **I**nsight & **S**tatistics.

Named to sit alongside the group's DNS solver, **RHEA** (*Reproducible
Hybrid-architecture flow solver Engineered for Academia*) — RHEA generates
the physics, METIS extracts the insight.

A reusable ML analysis layer sitting downstream of the group's DNS solver
for high-pressure transcritical flow. The DNS solver stays upstream and
authoritative; this framework standardizes ingestion, physics extraction,
and data-driven analysis of what it produces.

See `PROJECT_CONTEXT.md` for current status and active priorities, and
`docs/` for the full roadmap documents this repo implements.

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/generate_mock_dns.py --out data/mock/case_mock.h5
pytest -q
```

The mock-DNS generator produces a small synthetic HDF5 file with the
same layout the ingestion layer expects, so the pipeline runs and tests
pass before it's wired to real DNS output.

## Layout

```
src/metis/
    data/ingestion/     # HDF5 readers -> standardized DNSCase
    data/validation/
    data/preprocessing/
    data/datasets/
    features/
    models/             # baselines, classical_ml, lstm, fno, wno, deeponet
    training/
    evaluation/          # metrics, physics, ood, modal — filled in Phase 4-7
    registry/
    inference/
    monitoring/
    api/
    testing/            # mock DNS generator, shared test fixtures
```

## Status

Ingestion layer built and tested against synthetic mock DNS data.
Not yet wired to the group's real DNS output — see `PROJECT_CONTEXT.md`.

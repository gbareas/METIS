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

Ingestion (`HDF5Reader` for 3D snapshots, `SliceReader` for time-resolved
2D planes) is wired to, and cross-checked against, the group's real DNS
output — not just the synthetic mock data used in tests. The standard
physics layer (bulk dimensionless groups, spectra, POD) is built and
validated bit-for-bit against published Pub 4 results. The
regime-discovery reference task has been run end-to-end with a full
findings trail.

See `PROJECT_CONTEXT.md` for current priorities and `FINDINGS.md` for
results.

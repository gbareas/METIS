# Getting started

Target: install METIS, run the tests, run one analysis, and know where to
add a new module — in one sitting.

## 1. Install

```bash
git clone https://github.com/gbareas/METIS && cd METIS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # framework + tests + the `metis` command
```

Optional extras (see `pyproject.toml` for the rationale):

| extra | pulls in | needed for |
|---|---|---|
| `ml` | torch, scikit-learn, mlflow | autoencoder / trainer / experiment tracking |
| `report` | matplotlib | figures in `metis report` |
| `router` + `demo` | torch, anthropic, streamlit | the Track C/D router module |

## 2. Run the suite with synthetic data

No group data required — `metis.testing` generates DNS-shaped fixtures:

```bash
python scripts/generate_mock_dns.py --out data/mock/case_mock.h5
pytest -q
```

Tests that need real checkpoints / DNS / torch / an API key skip
themselves; the default run is self-contained.

## 3. Point METIS at real DNS data

METIS expects one data root holding the standard layout (see
[data_layout.md](data_layout.md)). Resolution order, first hit wins:

1. `--data-root /path` on any command
2. `data.root:` in a config file (`--config`, else `configs/default.yaml`)
3. `$METIS_DATA_ROOT`

```bash
export METIS_DATA_ROOT=/path/to/dns_data
metis cases list                     # what's registered
metis case validate case01           # structural + numerical checks
```

## 4. Run one analysis

```bash
metis analyze physics case01                         # bulk groups, wall Re_tau
metis analyze spectra case01 --slice s3_center --field u
metis analyze pod     case01 --slice s2_max_u --field u
metis analyze physics case01 --out artifacts/analysis/physics_case01
```

Every analysis returns the same `AnalysisResult` (name, inputs, config,
JSON-safe outputs, arrays, a validation summary, provenance) and can
`save` itself. Then turn one into a report:

```bash
metis report physics case01          # -> reports/physics-case01/{summary.md, metrics.json, figures/}
```

## 5. Reproduce the frozen benchmark

```bash
metis benchmark regime-v1            # writes results/regime_v1.json, exits non-zero on regression
```

See [reproducibility.md](reproducibility.md) for what makes runs
reproducible (config, dataset fingerprints, MLflow, the benchmark).

## 6. Where to add code

| you want to add… | see | put it in |
|---|---|---|
| a new standard analysis | [adding_an_analysis.md](adding_an_analysis.md) | `metis/analysis.py` |
| a new representation / ML model | [adding_a_model.md](adding_a_model.md) | `metis/models/` |
| a new physics feature | — | `metis/features/` (then expose via an analysis runner) |
| a new report type | — | `metis/reporting/generators.py` |
| a new CLI subcommand | — | `metis/cli.py` (`_cmd_*` + a subparser) |

The architecture and full module map: [architecture.md](architecture.md).

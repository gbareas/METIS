# Adding a standard analysis

A "standard analysis" is a named computation over a case that returns the
common `AnalysisResult` and is reachable as `metis analyze <kind> <case>`.
All four current ones (`physics`, `spectra`, `pod`, `regime_features`)
live in `src/metis/analysis.py` as ~15-line runner functions.

## 1. Write a runner

Signature: `_run_<name>(registry, descriptor, options: dict, validate: bool) -> _RunnerOutput`.

```python
def _run_wall_spectra(reg, desc, opts, validate):
    """1-D spectrum of a wall-adjacent slice. Options: slice_id, field."""
    slice_id = opts.get("slice_id", "s5_y_plus_10_cw")
    field = opts.get("field", "u")
    sc = desc.load_slice(slice_id)
    reports = [validate_slice_case(sc)] if validate else []

    k, e = wavenumber_spectrum(sc, field, "x")
    return _RunnerOutput(
        config={"slice_id": slice_id, "field": field},
        outputs={"n_wavenumbers": int(k.size), "peak_k": float(k[e.argmax()])},
        arrays={"k": k, "E": e},
        reports=reports,
    )
```

Rules:

- **Wrap an existing validated function** (`case_physics_summary`,
  `wavenumber_spectrum`, `pod`, …) — don't reimplement physics here.
- `outputs` must be JSON-safe scalars / short lists. Large arrays go in
  `arrays` (saved to `arrays.npz`).
- Load data through the descriptor (`desc.load()`, `desc.load_slice(id)`)
  — never build paths.
- Return validation reports; `run_analysis` merges them into
  `result.validation`. `validate=False` skips them.

## 2. Register it

```python
_RUNNERS = {
    ...,
    "wall_spectra": _run_wall_spectra,
}
```

`ANALYSES` and the CLI `choices` update automatically. Options are passed
through as keyword args, so add any new flags to `scripts/analyze.py` /
`metis.cli`'s `analyze` subparser and to `_ANALYSIS_OPTS` in `cli.py`.

## 3. Test it

`tests/unit/test_analysis.py` pattern: build a mock registry with
`mock_dns` / `mock_slices`, call `run_analysis(reg, "wall_spectra",
"case01", ...)`, assert on `result.outputs` / `result.arrays` /
`result.config`, and check `validate=False` yields `result.validation is
None`. Keep the test torch-free so it runs in CI.

## 4. (optional) A report generator

If it deserves a formatted write-up, add a `<name>_report(result_or_dict)
-> Report` to `metis/reporting/generators.py` and wire it into
`GENERATORS` / the `metis report` CLI. See
[adding_a_model.md](adding_a_model.md) and the existing generators for
the `Report` builder API.

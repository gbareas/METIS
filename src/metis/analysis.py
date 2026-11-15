"""Standard analysis API (milestone R6).

One entry point over the trusted physics functions::

    from metis.analysis import run_analysis
    result = run_analysis(registry, "pod", "case01", slice_id="s2_max_u", field="u")

`analysis` is one of `ANALYSES`; extra keyword options are analysis-
specific (see each runner's docstring). Every analysis returns the same
`AnalysisResult` — name, input case(s), the resolved config, JSON-safe
`outputs`, large `arrays`, a validation summary, and provenance — and
every result knows how to `save` / `load` itself.

This is deliberately a thin function-dispatch layer, not a class
hierarchy: the runners just adapt existing functions to the common shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metis.config import git_commit
from metis.data.datasets import build_feature_dataset
from metis.data.registry import CaseDescriptor, CaseRegistry
from metis.data.validation import validate_case, validate_dns_case, validate_slice_case
from metis.data.validation.report import ValidationReport
from metis.features.physics import (
    BULK_FIELDS,
    case_physics_summary,
    wall_normal_profiles,
)
from metis.features.pod import DEFAULT_ENERGY_THRESHOLD, pod
from metis.features.spectra import premultiplied_spectrum, wavenumber_spectrum

OUTPUTS_FILE = "outputs.json"
ARRAYS_FILE = "arrays.npz"


@dataclass
class _RunnerOutput:
    config: dict
    outputs: dict
    arrays: dict
    reports: list[ValidationReport]


@dataclass
class AnalysisResult:
    analysis: str
    case_ids: list[str]
    config: dict
    outputs: dict
    arrays: dict = field(default_factory=dict)
    validation: dict | None = None
    provenance: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = f"{self.analysis}({', '.join(self.case_ids)})"
        keys = ", ".join(self.outputs) or "-"
        val = "" if self.validation is None else (
            " [validation OK]" if self.validation["ok"]
            else f" [validation: {self.validation['errors']} error(s)]"
        )
        return f"{head}: outputs {{{keys}}}, arrays {list(self.arrays)}{val}"

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / OUTPUTS_FILE).write_text(json.dumps({
            "analysis": self.analysis,
            "case_ids": self.case_ids,
            "config": self.config,
            "outputs": self.outputs,
            "validation": self.validation,
            "provenance": self.provenance,
        }, indent=2))
        np.savez(out_dir / ARRAYS_FILE, **self.arrays)
        return out_dir

    @classmethod
    def load(cls, out_dir: str | Path) -> AnalysisResult:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / OUTPUTS_FILE).read_text())
        arrays = {}
        arrays_path = out_dir / ARRAYS_FILE
        if arrays_path.exists():
            with np.load(arrays_path) as npz:
                arrays = {k: npz[k] for k in npz.files}
        return cls(
            analysis=meta["analysis"], case_ids=meta["case_ids"],
            config=meta["config"], outputs=meta["outputs"], arrays=arrays,
            validation=meta.get("validation"), provenance=meta.get("provenance", {}),
        )


# --------------------------------------------------------------------- #
# runners: (registry, descriptor, options) -> _RunnerOutput
# --------------------------------------------------------------------- #
def _run_physics(reg, desc: CaseDescriptor, opts: dict, validate: bool) -> _RunnerOutput:
    """Bulk dimensionless groups + both wall friction Reynolds numbers,
    plus the wall-normal profiles of `BULK_FIELDS`. No options."""
    dns = desc.load()
    reports = [validate_dns_case(dns)] if validate else []
    summary = case_physics_summary(dns)
    profiles = wall_normal_profiles(dns, BULK_FIELDS)
    return _RunnerOutput(
        config={},
        outputs=summary,
        arrays={"y": dns.coordinates["y"], **profiles},
        reports=reports,
    )


def _run_spectra(reg, desc: CaseDescriptor, opts: dict, validate: bool) -> _RunnerOutput:
    """1D wavenumber spectrum E(k) and premultiplied k*E(k). Options:
    `slice_id` (default s3_center), `field` (default u), `axis` (x|z)."""
    slice_id = opts.get("slice_id", "s3_center")
    fld = opts.get("field", "u")
    axis = opts.get("axis", "x")
    sc = desc.load_slice(slice_id)
    reports = [validate_slice_case(sc)] if validate else []
    k, e = wavenumber_spectrum(sc, fld, axis)
    _, ke = premultiplied_spectrum(sc, fld, axis)
    dk = float(k[1] - k[0])
    lhs, rhs = float(np.sum(e) * dk), float(np.mean(sc.snapshots[fld] ** 2))
    return _RunnerOutput(
        config={"slice_id": slice_id, "field": fld, "axis": axis},
        outputs={
            "n_wavenumbers": int(k.size),
            "parseval_lhs": lhs, "parseval_rhs": rhs,
            "parseval_rel_error": abs(lhs - rhs) / rhs if rhs else 0.0,
        },
        arrays={"k": k, "E": e, "kE": ke},
        reports=reports,
    )


def _run_pod(reg, desc: CaseDescriptor, opts: dict, validate: bool) -> _RunnerOutput:
    """Snapshot POD of a slice field. Options: `slice_id` (default
    s3_center), `field` (default u), `energy_threshold` (default 0.99)."""
    slice_id = opts.get("slice_id", "s3_center")
    fld = opts.get("field", "u")
    thr = float(opts.get("energy_threshold", DEFAULT_ENERGY_THRESHOLD))
    sc = desc.load_slice(slice_id)
    reports = [validate_slice_case(sc)] if validate else []
    res = pod(sc, fld, thr)
    return _RunnerOutput(
        config={"slice_id": slice_id, "field": fld, "energy_threshold": thr},
        outputs={
            "rank": int(res.singular_values.size),
            "energy_captured": float(res.energy_captured),
            "energy_threshold": float(res.energy_threshold),
            "n_snapshots": int(res.coefficients.shape[1]),
        },
        arrays={
            "singular_values": res.singular_values,
            "singular_values_all": res.singular_values_all,
            "energy_fractions": res.energy_fractions(),
            "leading_mode": res.mode_field(0),
        },
        reports=reports,
    )


def _run_regime_features(reg, desc: CaseDescriptor, opts: dict, validate: bool) -> _RunnerOutput:
    """The regime-discovery feature vector for one case. Options:
    `feature_set` (compact|rich|bulk|mean_profile|rms_profile|pod)."""
    feature_set = opts.get("feature_set", "compact")
    ds = build_feature_dataset(reg, [desc.case_id], feature_set)
    reports = [validate_case(desc)] if validate else []
    return _RunnerOutput(
        config={"feature_set": feature_set},
        outputs={"feature_set": feature_set, "n_features": int(ds.X.shape[1]),
                 "feature_names": ds.feature_names},
        arrays={"features": ds.X[0]},
        reports=reports,
    )


_RUNNERS = {
    "physics": _run_physics,
    "spectra": _run_spectra,
    "pod": _run_pod,
    "regime_features": _run_regime_features,
}
ANALYSES = tuple(_RUNNERS)


def _merge_reports(reports: list[ValidationReport]) -> ValidationReport | None:
    reports = [r for r in reports if r is not None]
    if not reports:
        return None
    merged = ValidationReport(subject=" + ".join(r.subject for r in reports))
    for r in reports:
        merged.extend(r)
    return merged


def _report_to_dict(report: ValidationReport) -> dict:
    return {
        "ok": report.ok,
        "errors": len(report.errors),
        "warnings": len(report.warnings),
        "issues": [
            {"severity": i.severity, "check": i.check, "message": i.message}
            for i in report.issues
        ],
    }


def run_analysis(
    registry: CaseRegistry,
    analysis: str,
    case: str,
    *,
    validate: bool = True,
    strict: bool = True,
    **options,
) -> AnalysisResult:
    """Run `analysis` on `case` and return a standard `AnalysisResult`.

    With `validate=True` (default) the data validators run and, if
    `strict` is also True (default), a validation **error** aborts the
    analysis with `ValidationError` — no artifact is produced from
    known-invalid input. `strict=False` proceeds anyway and stamps
    `provenance["validation_overridden"] = True`. `validate=False` skips
    validation entirely.
    """
    if analysis not in _RUNNERS:
        raise ValueError(
            f"unknown analysis {analysis!r}; choose from {list(ANALYSES)}"
        )
    descriptor = registry[case]  # KeyError (listing known cases) if unknown
    out = _RUNNERS[analysis](registry, descriptor, options, validate)
    merged = _merge_reports(out.reports)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "data_root": str(registry.data_root),
    }
    if merged is not None and not merged.ok:
        if strict:
            merged.raise_if_failed()  # ValidationError, no result returned
        provenance["validation_overridden"] = True

    return AnalysisResult(
        analysis=analysis,
        case_ids=[case],
        config=out.config,
        outputs=out.outputs,
        arrays=out.arrays,
        validation=_report_to_dict(merged) if merged is not None else None,
        provenance=provenance,
    )

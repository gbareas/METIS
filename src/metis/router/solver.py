"""Precomputed full-solver fallback (Track C/D milestone M2).

This is a lookup against the group's existing DNS database, never a live
solve — there is no HPC/Slurm connection here and none is planned for the
MVP (see docs/agent_implementation_plan.md's Definition of Done /
Explicitly Out of Scope). It can only answer for the 11 cases that
already have DNS output (case01-09, case10, case15 — not "15 cases";
case11-14 were never simulated), looked up by matching
`(Pb_Pc, Thw_Tc, Tcw_Tc)` against `pub4_grassmann_rom/results/
case_descriptors.json`.

Returns the group's converged (solver-level, not finite-ensemble) RMS
fields for u', T', cp' on the same `xy_slice_1` plane and in the same
`{"fields": ..., "field_means": ...}` shape as
`metis.router.surrogate.surrogate_infer`, so the two are directly
comparable — that comparability is the entire point of the router.
"""
from __future__ import annotations

from neuralop_bench.data import FIELDS, SliceCase, load_descriptors

SLICE_ID = "xy_slice_1"
MATCH_TOLERANCE = 1e-6


def _find_case(case_params: dict) -> int:
    """Match `case_params` against the known case_descriptors table.

    Requires a (near-)exact match — this is a lookup, not a
    nearest-neighbor guess. Silently substituting a "close enough" case
    would misrepresent a genuinely different physical point as solved,
    which is exactly what M2's "never a live solve" constraint rules out.
    """
    descriptors = load_descriptors()
    for case_id, desc in descriptors.items():
        if all(
            abs(case_params[k] - desc[k]) <= MATCH_TOLERANCE
            for k in ("Pb_Pc", "Thw_Tc", "Tcw_Tc")
        ):
            return int(case_id.removeprefix("case"))
    raise KeyError(
        f"No precomputed DNS case matches {case_params} within "
        f"{MATCH_TOLERANCE} — solver_lookup only covers the group's 11 "
        f"already-simulated cases (case01-09, case10, case15), it cannot "
        f"solve for a novel operating point."
    )


def solver_lookup(case_params: dict) -> dict:
    """Precomputed full-solver answer for `case_params = {"Pb_Pc": ..,
    "Thw_Tc": .., "Tcw_Tc": ..}`, if it matches one of the 11
    already-simulated cases; raises `KeyError` otherwise."""
    case = _find_case(case_params)
    sc = SliceCase.load(case, SLICE_ID, None, load_descriptors())
    if sc.conv_rms is None:
        raise ValueError(f"case{case:02d}: no converged RMS data on slice {SLICE_ID}")

    fields = {name: sc.conv_rms[i] for i, name in enumerate(FIELDS)}
    return {
        "case": case,
        "fields": fields,
        "field_means": {name: float(arr.mean()) for name, arr in fields.items()},
    }

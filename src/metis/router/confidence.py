"""Rule-based OOD/confidence diagnostic for the router agent (Track C/D,
docs/agent_implementation_plan.md, milestone M1).

This reuses the *validated* finding from pub5_neural_operators
(FINDINGS.md §5.13, "H4 OOD"), not a new metric: blind OOD error on the
task-A2 checkpoints is architecture-independent (spread ~0.05 across 6
architectures) and tracks the cold-wall temperature ratio `Tcw_Tc`
specifically — not distance in the surrogate's own conditioning space
`(Pb_Pc, Thw_Tc)`. Both held-out cases sit *inside* the training
`(Pb_Pc, Thw_Tc)` envelope, yet have very different outcomes:

    case15: Tcw_Tc=0.982 (mild excursion, still subcritical) -> T' err ~0.37
    case10: Tcw_Tc=1.035 (crosses the pseudo-critical boundary)-> T' err ~0.92 (collapse)

A different diagnostic (latent-geometry vs. Pub 4's Grassmann principal
angles, `pub5_neural_operators/src/neuralop_bench/xai/latent_geometry.py`)
was tried and explicitly failed (FINDINGS §5.17: correlation with OOD
skill ~= -0.13) — it is deliberately NOT used here.

This rests on n=2 OOD cases (FINDINGS §5.13: "suggestive, not proven").
The routing policy below is conservative *because* of that small n: only
`"high"` confidence (fully inside every training range) is trusted with
the surrogate; both `"medium"` and `"low"` fall back to the full solver.
`"medium"` is kept as a distinct level for the explanation the agent
gives (M3), not because it is currently routed differently from `"low"`.
"""
from __future__ import annotations

from dataclasses import dataclass

# Observed range per ratio across the case01-09 training grid
# (pub4_grassmann_rom/results/case_descriptors.json).
TRAINING_RANGES = {
    "Pb_Pc": (1.5, 5.0),
    "Thw_Tc": (1.1, 1.4),
    "Tcw_Tc": (0.80, 0.95),
}

# T/T_critical = 1.0 is the pseudo-critical boundary; RHEA's training grid
# never crosses it at the cold wall (max Tcw_Tc = 0.95).
PSEUDOCRITICAL_RATIO = 1.0

CONFIDENT_LEVELS = frozenset({"high"})


@dataclass
class Confidence:
    level: str  # "high" | "medium" | "low"
    reason: str
    in_envelope: bool
    Tcw_Tc: float

    def trusts_surrogate(self) -> bool:
        return self.level in CONFIDENT_LEVELS


def confidence_score(case_params: dict) -> Confidence:
    """Confidence that the wrapped surrogate is reliable for
    `case_params = {"Pb_Pc": float, "Thw_Tc": float, "Tcw_Tc": float}`.
    """
    missing = [k for k in ("Pb_Pc", "Thw_Tc", "Tcw_Tc") if k not in case_params]
    if missing:
        raise ValueError(f"confidence_score requires {missing}")

    tcw = case_params["Tcw_Tc"]
    _tcw_lo, tcw_hi = TRAINING_RANGES["Tcw_Tc"]
    in_envelope = all(
        TRAINING_RANGES[k][0] <= case_params[k] <= TRAINING_RANGES[k][1]
        for k in ("Pb_Pc", "Thw_Tc", "Tcw_Tc")
    )

    if tcw >= PSEUDOCRITICAL_RATIO:
        return Confidence(
            level="low",
            reason=(
                f"Tcw_Tc={tcw:.3f} crosses the pseudo-critical boundary "
                f"(>= {PSEUDOCRITICAL_RATIO}) — matches case10's collapse "
                f"(blind OOD T' error ~0.92)."
            ),
            in_envelope=in_envelope,
            Tcw_Tc=tcw,
        )
    if tcw > tcw_hi:
        return Confidence(
            level="medium",
            reason=(
                f"Tcw_Tc={tcw:.3f} exceeds the training envelope max "
                f"({tcw_hi}) but stays subcritical — matches case15's mild "
                f"excursion (blind OOD T' error ~0.37, close to in-distribution)."
            ),
            in_envelope=in_envelope,
            Tcw_Tc=tcw,
        )
    if in_envelope:
        return Confidence(
            level="high",
            reason="Pb_Pc, Thw_Tc, and Tcw_Tc all fall inside the training envelope.",
            in_envelope=in_envelope,
            Tcw_Tc=tcw,
        )
    return Confidence(
        level="medium",
        reason=(
            "Tcw_Tc is within its training range, but Pb_Pc or Thw_Tc falls "
            "outside its training range — an untested combination even "
            "though the specific OOD failure mode found so far (§5.13) was "
            "Tcw_Tc-driven, not Pb_Pc/Thw_Tc-driven."
        ),
        in_envelope=in_envelope,
        Tcw_Tc=tcw,
    )

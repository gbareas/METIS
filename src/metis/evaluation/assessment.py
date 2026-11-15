"""The "is it actually better?" gate (milestone I5, design rule §17).

    A model cannot be called better because one generic ML metric
    improves — the relevant physical diagnostics must agree.

`assess_model` compares a candidate's metrics to a baseline's, split into
an ML group and a physical group, and only returns `is_better=True` when
the candidate improves at least one ML metric AND regresses no physical
diagnostic (beyond `margin`).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Metric name fragments where a *lower* value is better (error/distance
# metrics). Everything else is treated as higher-is-better.
COMMON_LOWER_IS_BETTER = (
    "mae", "rmse", "nrmse", "relative_l2", "rel_l2", "_l1",
    "rel_shift", "_delta", "_disparity", "error",
)


def is_lower_better(name: str, lower_is_better: Iterable[str] = COMMON_LOWER_IS_BETTER) -> bool:
    return any(frag in name for frag in lower_is_better)


@dataclass
class ImprovementVerdict:
    ml_deltas: dict[str, float]
    physical_deltas: dict[str, float]
    ml_improved: list[str]
    ml_regressed: list[str]
    physical_improved: list[str]
    physical_regressed: list[str]
    is_better: bool
    reason: str


def _classify(
    candidate: dict, baseline: dict, names: Iterable[str], margin: float,
    lower_is_better: Iterable[str],
) -> tuple[dict, list[str], list[str]]:
    deltas, improved, regressed = {}, [], []
    for name in names:
        if name not in candidate or name not in baseline:
            continue
        raw = float(candidate[name]) - float(baseline[name])
        # signed so that positive always means "better"
        signed = -raw if is_lower_better(name, lower_is_better) else raw
        deltas[name] = signed
        if signed > margin:
            improved.append(name)
        elif signed < -margin:
            regressed.append(name)
    return deltas, improved, regressed


def assess_model(
    candidate: dict,
    baseline: dict,
    *,
    ml_metrics: Iterable[str],
    physical_metrics: Iterable[str],
    margin: float = 1e-6,
    lower_is_better: Iterable[str] = COMMON_LOWER_IS_BETTER,
) -> ImprovementVerdict:
    ml_d, ml_up, ml_down = _classify(candidate, baseline, ml_metrics, margin, lower_is_better)
    ph_d, ph_up, ph_down = _classify(
        candidate, baseline, physical_metrics, margin, lower_is_better
    )

    is_better = bool(ml_up) and not ph_down
    if not ml_up:
        reason = "no ML metric improved"
    elif ph_down:
        reason = f"ML improved ({ml_up}) but physical diagnostics regressed ({ph_down})"
    else:
        reason = f"ML improved ({ml_up}) with no physical regression"

    return ImprovementVerdict(
        ml_deltas=ml_d, physical_deltas=ph_d,
        ml_improved=ml_up, ml_regressed=ml_down,
        physical_improved=ph_up, physical_regressed=ph_down,
        is_better=is_better, reason=reason,
    )

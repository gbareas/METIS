"""Curated router scenarios, kept free of torch/anthropic imports so the
confidence policy can be demoed without the router extra (`metis demo`).
See `metis.router.demo` for the provenance of every value.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    description: str
    Pb_Pc: float
    Thw_Tc: float
    Tcw_Tc: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("case01", "Training grid: lowest pressure, mildest wall heating", 1.5, 1.1, 0.95),
    Scenario("case03", "Training grid: lowest pressure, strongest wall heating", 1.5, 1.4, 0.80),
    Scenario("case05", "Training grid center", 2.0, 1.2, 0.90),
    Scenario("case07", "Training grid: highest pressure, mildest wall heating", 5.0, 1.1, 0.95),
    Scenario("case09", "Training grid: highest pressure, strongest wall heating", 5.0, 1.4, 0.80),
    Scenario(
        "case10 (OOD)",
        "Cold wall crosses into supercritical — the surrogate's known collapse case",
        1.5, 1.185, 1.035,
    ),
    Scenario(
        "case15 (OOD)",
        "Mild thermal excursion beyond the training envelope",
        1.5, 1.132, 0.982,
    ),
    Scenario(
        "novel point",
        "Untested operating point — no DNS case in the database matches it",
        2.5, 1.15, 1.05,
    ),
)


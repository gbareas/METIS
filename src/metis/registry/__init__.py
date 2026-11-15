"""Artifact registry (milestone I8) — which datasets / models exist and
whether they are approved for reuse.

    from metis.registry import ArtifactRegistry
    reg = ArtifactRegistry("artifacts")
    reg.register("regime_compact_v1", "dataset", metrics={"n_cases": 11})

Not to be confused with `metis.data.registry` (the DNS *case* registry).
"""
from metis.registry.artifact_registry import (
    KINDS,
    STATUSES,
    ArtifactEntry,
    ArtifactRegistry,
)

__all__ = ["KINDS", "STATUSES", "ArtifactEntry", "ArtifactRegistry"]

"""Evaluation layer.

- `metrics`        — generic pointwise metrics (MAE / RMSE / nRMSE / rel-L2 / R²)
- `physical`      — physical-diagnostic agreement (profiles, spectra, POD energy)
- `representation` — latent organisation vs the regime axes, stability, correlations
- `assessment`   — the "is it actually better?" gate (ML metric up AND no physical regression)
- `regime`       — the regime-discovery evaluation
- `benchmark`     — the frozen regime-v1 benchmark
"""
from metis.evaluation.assessment import assess_model
from metis.evaluation.metrics import pointwise_metrics
from metis.evaluation.physical import physical_field_report

__all__ = ["assess_model", "physical_field_report", "pointwise_metrics"]

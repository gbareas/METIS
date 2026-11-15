"""Reporting layer (milestone I6).

    from metis.reporting import Report
    from metis.reporting.generators import regime_v1_report

Turns an experiment result into `<out>/{summary.md, metrics.json,
figures/}`. Figures need the `report` extra (matplotlib); the text and
JSON are always produced.
"""
from metis.reporting.report import Report, has_matplotlib

__all__ = ["Report", "has_matplotlib"]

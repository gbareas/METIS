"""Data validation layer (milestone R3).

    from metis.data.validation import validate_case, validate_dns_case

`validate_*` functions return a `ValidationReport`; call
`report.raise_if_failed()` to turn errors into a `ValidationError`.
"""
from metis.data.validation.checks import (
    validate_case,
    validate_compatibility,
    validate_dns_case,
    validate_slice_case,
)
from metis.data.validation.report import (
    ValidationError,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_case",
    "validate_compatibility",
    "validate_dns_case",
    "validate_slice_case",
]

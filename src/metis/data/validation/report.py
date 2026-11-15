"""Structured validation diagnostics (milestone R3).

Validators return a `ValidationReport` — a list of `ValidationIssue`s with
severities — rather than a bare bool, so a caller can log every problem,
decide whether warnings are tolerable, or fail hard with
`report.raise_if_failed()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # ERROR | WARNING
    check: str     # short slug, e.g. "grid_matches_metadata"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    subject: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True if no errors (warnings are allowed)."""
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)

    def summary(self) -> str:
        head = (
            f"{self.subject}: OK"
            if self.ok
            else f"{self.subject}: {len(self.errors)} error(s)"
        )
        if self.warnings:
            head += f", {len(self.warnings)} warning(s)"
        lines = [head] + [f"  {i}" for i in self.issues]
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError(self)


class ValidationError(Exception):
    """Raised by `ValidationReport.raise_if_failed` when errors are present."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


class Checks:
    """Small builder a validator accumulates issues into."""

    def __init__(self, subject: str):
        self._report = ValidationReport(subject=subject)

    def error(self, check: str, message: str) -> None:
        self._report.issues.append(ValidationIssue(ERROR, check, message))

    def warn(self, check: str, message: str) -> None:
        self._report.issues.append(ValidationIssue(WARNING, check, message))

    def expect(self, condition: bool, check: str, message: str) -> bool:
        """Record an error unless `condition` holds. Returns `condition`
        so callers can skip dependent checks."""
        if not condition:
            self.error(check, message)
        return bool(condition)

    def report(self) -> ValidationReport:
        return self._report

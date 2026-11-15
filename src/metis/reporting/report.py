"""Report builder (milestone I6).

A `Report` accumulates markdown sections, a flat metrics dict, and named
figures, then `write`s `<out_dir>/{summary.md, metrics.json, figures/}`.
Figures need matplotlib (the `report` extra); without it the text and
JSON are still produced and `summary.md` notes the skipped figures.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from metis import __version__
from metis.config import git_commit

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the report extra
    _plt = None
    _MPL_AVAILABLE = False

SUMMARY_FILE = "summary.md"
METRICS_FILE = "metrics.json"
FIGURES_DIR = "figures"


def has_matplotlib() -> bool:
    return _MPL_AVAILABLE


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _md_table(rows: Sequence[dict]) -> str:
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


@dataclass
class _Section:
    heading: str
    body: str = ""
    table: list[dict] | None = None


@dataclass
class Report:
    title: str
    sections: list[_Section] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    figures: dict[str, Callable] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def section(self, heading: str, body: str = "", table: Sequence[dict] | None = None) -> Report:
        self.sections.append(_Section(heading, body.strip(), list(table) if table else None))
        return self

    def add_metrics(self, metrics: dict, prefix: str = "") -> Report:
        for k, v in metrics.items():
            self.metrics[f"{prefix}{k}"] = v
        return self

    def add_figure(self, name: str, draw: Callable) -> Report:
        """`draw(fig)` populates a matplotlib Figure (ignored if
        matplotlib is unavailable)."""
        self.figures[name] = draw
        return self

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        (out_dir / FIGURES_DIR).mkdir(parents=True, exist_ok=True)

        prov = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metis_version": __version__,
            "git_commit": git_commit(),
            **self.provenance,
        }

        written_figs: list[str] = []
        if self.figures and _MPL_AVAILABLE:
            for name, draw in self.figures.items():
                fig = _plt.figure(figsize=(7, 4))
                draw(fig)
                fig.tight_layout()
                fig.savefig(out_dir / FIGURES_DIR / f"{name}.png", dpi=120)
                _plt.close(fig)
                written_figs.append(name)

        lines = [f"# {self.title}", ""]
        lines += [f"- **{k}**: {v}" for k, v in prov.items()] + [""]
        for s in self.sections:
            lines += [f"## {s.heading}", ""]
            if s.body:
                lines += [s.body, ""]
            if s.table:
                lines += [_md_table(s.table), ""]
        if self.figures:
            lines += ["## Figures", ""]
            for name in self.figures:
                if name in written_figs:
                    lines.append(f"![{name}]({FIGURES_DIR}/{name}.png)")
                else:
                    lines.append(f"_{name}: skipped (matplotlib not installed — `pip install -e \".[report]\"`)_")
            lines.append("")

        (out_dir / SUMMARY_FILE).write_text("\n".join(lines))
        (out_dir / METRICS_FILE).write_text(
            json.dumps({"title": self.title, "provenance": prov, "metrics": self.metrics}, indent=2)
        )
        return out_dir

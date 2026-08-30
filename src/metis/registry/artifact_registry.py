"""Directory-backed registry of validated datasets and models
(milestone I8).

One JSON file, `artifacts/registry.json`, listing what has been produced
and whether it is approved for reuse. Deliberately not a service, a
database, or a model store — just enough to answer "which dataset / model
should I reuse, and is it trustworthy?".

`status` lifecycle: `experimental` -> `validated` -> `deprecated`.
"validated" means *approved for reuse under a documented scope*, not
deployed anywhere.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from metis import __version__
from metis.config import git_commit

KINDS = ("dataset", "model", "report")
STATUSES = ("experimental", "validated", "deprecated")
REGISTRY_FILE = "registry.json"


@dataclass
class ArtifactEntry:
    id: str
    kind: str                       # dataset | model | report
    status: str = "experimental"
    created_at: str = ""
    updated_at: str = ""
    git_commit: str = ""
    metis_version: str = ""
    run_id: str | None = None       # MLflow run that produced it
    dataset_id: str | None = None   # for models: which dataset it was fit on
    path: str | None = None         # location of the artifact, relative to the registry root
    scope: str | None = None        # documented reuse scope (required to validate)
    metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")


class ArtifactRegistry:
    def __init__(self, root: str | Path = "artifacts"):
        self.root = Path(root)
        self.path = self.root / REGISTRY_FILE
        self._entries: dict[str, ArtifactEntry] = {}
        if self.path.exists():
            self._load()

    # -- persistence -----------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._entries = {e["id"]: ArtifactEntry(**e) for e in data.get("entries", [])}

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(e) for e in self._entries.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)
        return self.path

    # -- access --------------------------------------------------
    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, artifact_id: str) -> ArtifactEntry:
        try:
            return self._entries[artifact_id]
        except KeyError:
            raise KeyError(
                f"no artifact {artifact_id!r} (have: {sorted(self._entries)})"
            ) from None

    def get(self, artifact_id: str, default=None):
        return self._entries.get(artifact_id, default)

    def list(self, *, kind: str | None = None, status: str | None = None) -> list[ArtifactEntry]:
        out = list(self._entries.values())
        if kind is not None:
            out = [e for e in out if e.kind == kind]
        if status is not None:
            out = [e for e in out if e.status == status]
        return sorted(out, key=lambda e: e.id)

    # -- mutation -----------------------------------------------
    def register(
        self,
        artifact_id: str,
        kind: str,
        *,
        status: str = "experimental",
        run_id: str | None = None,
        dataset_id: str | None = None,
        path: str | Path | None = None,
        scope: str | None = None,
        metrics: dict | None = None,
    ) -> ArtifactEntry:
        """Add a new artifact. Raises if `artifact_id` already exists — use
        `update` to change an existing entry."""
        if artifact_id in self._entries:
            raise ValueError(f"artifact {artifact_id!r} already registered; use update()")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry = ArtifactEntry(
            id=artifact_id, kind=kind, status=status,
            created_at=now, updated_at=now,
            git_commit=git_commit(), metis_version=__version__,
            run_id=run_id, dataset_id=dataset_id,
            path=None if path is None else str(path),
            scope=scope, metrics=dict(metrics or {}),
        )
        self._entries[artifact_id] = entry
        self.save()
        return entry

    def update(self, artifact_id: str, **fields) -> ArtifactEntry:
        entry = self[artifact_id]
        for key, value in fields.items():
            if not hasattr(entry, key) or key in ("id", "created_at"):
                raise ValueError(f"cannot update field {key!r}")
            setattr(entry, key, value)
        entry.__post_init__()  # re-validate kind/status
        entry.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        return entry

    def set_status(self, artifact_id: str, status: str) -> ArtifactEntry:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        entry = self[artifact_id]
        if status == "validated" and not entry.scope:
            raise ValueError(
                f"{artifact_id!r} needs a documented `scope` before it can be validated"
            )
        return self.update(artifact_id, status=status)

    def promote(self, artifact_id: str) -> ArtifactEntry:
        """experimental -> validated (requires a `scope`)."""
        return self.set_status(artifact_id, "validated")

    def deprecate(self, artifact_id: str) -> ArtifactEntry:
        return self.set_status(artifact_id, "deprecated")

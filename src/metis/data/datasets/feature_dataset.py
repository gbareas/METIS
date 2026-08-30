"""Level-1 feature dataset: one row of physical features per DNS case
(milestone R5).

An extracted, cached artifact that experiments consume instead of
re-reading raw DNS every time a metric changes. Stored as two files in
one directory::

    <dir>/data.npz         X (n_cases, n_features)
    <dir>/metadata.json    case_ids, feature_names, feature_set, provenance

`provenance` carries the dataset id, source cases, feature set, generation
config, code version, and a fingerprint used for cache validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_FILE = "data.npz"
METADATA_FILE = "metadata.json"


@dataclass
class FeatureDataset:
    X: np.ndarray                      # (n_cases, n_features), float64
    case_ids: list[str]
    feature_names: list[str]
    feature_set: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.case_ids = list(self.case_ids)
        self.feature_names = list(self.feature_names)
        n, m = self.X.shape
        if n != len(self.case_ids):
            raise ValueError(f"X has {n} rows but {len(self.case_ids)} case_ids")
        if m != len(self.feature_names):
            raise ValueError(f"X has {m} cols but {len(self.feature_names)} feature_names")

    # -- access ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.case_ids)

    @property
    def fingerprint(self) -> str | None:
        return self.provenance.get("fingerprint")

    def row(self, case_id: str) -> np.ndarray:
        try:
            return self.X[self.case_ids.index(case_id)]
        except ValueError:
            raise KeyError(
                f"{case_id!r} not in dataset (has: {self.case_ids})"
            ) from None

    def matrix(self, case_ids: list[str] | None = None) -> np.ndarray:
        """The feature matrix, optionally subset/reordered to `case_ids`."""
        if case_ids is None:
            return self.X
        idx = [self.case_ids.index(c) for c in case_ids]  # KeyError-ish via ValueError
        return self.X[idx]

    # -- persistence ----------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / DATA_FILE, X=self.X)
        (out_dir / METADATA_FILE).write_text(
            json.dumps(
                {
                    "feature_set": self.feature_set,
                    "case_ids": self.case_ids,
                    "feature_names": self.feature_names,
                    "provenance": self.provenance,
                },
                indent=2,
            )
        )
        return out_dir

    @classmethod
    def load(cls, out_dir: str | Path) -> FeatureDataset:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / METADATA_FILE).read_text())
        with np.load(out_dir / DATA_FILE) as npz:
            X = npz["X"]
        return cls(
            X=X,
            case_ids=meta["case_ids"],
            feature_names=meta["feature_names"],
            feature_set=meta["feature_set"],
            provenance=meta.get("provenance", {}),
        )

    @staticmethod
    def exists_at(out_dir: str | Path) -> bool:
        out_dir = Path(out_dir)
        return (out_dir / DATA_FILE).exists() and (out_dir / METADATA_FILE).exists()

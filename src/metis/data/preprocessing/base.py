"""Reusable, serialisable preprocessing transforms (milestone R4).

A `Transform` is `fit` once (on training data only) then `transform`
applied to any split — the fit/apply split is what keeps held-out cases
genuinely unseen. Fitted transforms serialise to plain JSON via
`save` / `load`.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import ClassVar


class NotFittedError(RuntimeError):
    """Raised when `transform` is called before `fit`."""


class Transform(abc.ABC):
    """Base class: `fit(X) -> self`, `transform(X) -> X'`.

    Subclasses register themselves by class name so `load` can rebuild
    them. `get_params` must return JSON-safe values; `from_params` is its
    inverse.
    """

    _registry: ClassVar[dict[str, type[Transform]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Transform._registry[cls.__name__] = cls

    @abc.abstractmethod
    def fit(self, X) -> Transform:
        ...

    @abc.abstractmethod
    def transform(self, X):
        ...

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        raise NotImplementedError(f"{type(self).__name__} has no inverse_transform")

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        ...

    @abc.abstractmethod
    def get_params(self) -> dict:
        ...

    @classmethod
    @abc.abstractmethod
    def from_params(cls, params: dict) -> Transform:
        ...

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__}.fit(...) must be called first")


def to_dict(transform: Transform) -> dict:
    return {"type": type(transform).__name__, "params": transform.get_params()}


def from_dict(data: dict) -> Transform:
    try:
        cls = Transform._registry[data["type"]]
    except KeyError:
        raise ValueError(f"unknown transform type {data.get('type')!r}") from None
    return cls.from_params(data["params"])


def save(transform: Transform, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_dict(transform), indent=2))
    return path


def load(path: str | Path) -> Transform:
    return from_dict(json.loads(Path(path).read_text()))

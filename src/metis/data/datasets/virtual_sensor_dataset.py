"""Level-3 virtual-sensor temporal dataset (Phase C / C2).

Turns the `processed_slices` product into a multivariate **sample-step**
time series per `docs/virtual_sensor_protocol.md` (frozen C1): a handful
of physically-placed XZ-slice probes, each reduced to plane-aggregate
scalar channels per snapshot, then chronologically split and windowed for
forecasting.

Stored under one directory as three `.npz` splits + a `metadata.json`::

    <dir>/train.npz  X (n_steps, n_ch) float32 raw channel values,
                     case_ids (n_steps,), step_idx (n_steps,),
                     window_anchors (n_windows,) int
    <dir>/val.npz
    <dir>/ood.npz
    <dir>/metadata.json  channels, probes, context_length, horizons,
                         scaler {mean,std} (fit on train), static_context,
                         splits, provenance

Channels are **raw**; `metadata["scaler"]` (per-channel mean/std fit on
the train split only) is how to standardise. A window anchored at row `t`
has context rows `t-L+1 .. t` and target rows `t+h` for each horizon `h`;
anchors never cross a case boundary or the train/val step boundary
(each split is built from its own step slice).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metis.config import git_commit
from metis.data.ingestion import slice_reader as _slice_reader
from metis.data.registry import CaseRegistry

_SPLITS = ("train", "val", "ood")
METADATA_FILE = "metadata.json"
_FIELDS = ("u", "T", "cp")
_AGGREGATES = ("pmf", "rms")          # plane-mean fluctuation, plane RMS fluctuation
_THIS = Path(__file__)


# --------------------------------------------------------------------- #
def compute_sensor_channels(
    fluctuations: dict[str, np.ndarray],
    *,
    fields: tuple[str, ...] = _FIELDS,
    aggregates: tuple[str, ...] = _AGGREGATES,
) -> tuple[np.ndarray, list[str]]:
    """One probe -> `(n_snapshots, len(fields)*len(aggregates))` channel
    matrix. `fluctuations[field]` is `(n, NX, NZ)` (mean already removed).

    - `pmf` = plane-mean fluctuation  <f'>_xz(t)
    - `rms` = plane RMS fluctuation   sqrt(<f'^2>_xz)(t)
    """
    cols, names = [], []
    for f in fields:
        fp = np.asarray(fluctuations[f], dtype=np.float64)
        flat = fp.reshape(len(fp), -1)
        for agg in aggregates:
            if agg == "pmf":
                cols.append(flat.mean(axis=1))
            elif agg == "rms":
                cols.append(np.sqrt((flat**2).mean(axis=1)))
            else:
                raise ValueError(f"unknown aggregate {agg!r}")
            names.append(f"{f}_{agg}")
    return np.column_stack(cols).astype(np.float32), names


def normalize_vs_split_config(cfg: dict) -> dict:
    """`splits.<name>` yaml shape -> `{split: {"cases": [...],
    "steps": (start, stop)}}`."""
    tc = list(cfg["train_cases"])
    return {
        "train": {"cases": tc, "steps": tuple(cfg["train_steps"])},
        "val": {"cases": tc, "steps": tuple(cfg["val_steps"])},
        "ood": {"cases": list(cfg["ood_cases"]), "steps": tuple(cfg["ood_steps"])},
    }


# --------------------------------------------------------------------- #
@dataclass
class VirtualSensorDataset:
    X: np.ndarray                 # (n_steps, n_ch) raw channel values
    case_ids: np.ndarray          # (n_steps,) str
    step_idx: np.ndarray          # (n_steps,) int, index within that case's slice
    window_anchors: np.ndarray    # (n_windows,) int row indices t (last context step)
    channels: list[str]
    context_length: int
    horizons: list[int]
    split: str
    scaler: dict                  # {"mean": [n_ch], "std": [n_ch]}, fit on train
    static_context: dict          # {case_id: {"Pb_Pc":.., "Thw_Tc":.., "Tcw_Tc":.., "y_loc": {..}}}
    provenance: dict

    def __len__(self) -> int:
        return len(self.window_anchors)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def standardize(self, X=None) -> np.ndarray:
        X = self.X if X is None else X
        m = np.asarray(self.scaler["mean"], dtype=np.float32)
        s = np.asarray(self.scaler["std"], dtype=np.float32)
        return (np.asarray(X, dtype=np.float32) - m) / s

    def inverse(self, Xs) -> np.ndarray:
        m = np.asarray(self.scaler["mean"], dtype=np.float32)
        s = np.asarray(self.scaler["std"], dtype=np.float32)
        return np.asarray(Xs, dtype=np.float32) * s + m

    def assemble(self, *, standardize: bool = True):
        """Materialise the windows.

        Returns `(Xc, Y, anchor_case, anchor_step)` where
        `Xc` is `(n_windows, context_length, n_ch)`,
        `Y`  is `(n_windows, len(horizons), n_ch)`.
        """
        data = self.standardize() if standardize else np.asarray(self.X, dtype=np.float32)
        lgth = self.context_length
        anchors = self.window_anchors
        hz = np.asarray(self.horizons)
        ctx_off = np.arange(-lgth + 1, 1)
        Xc = data[anchors[:, None] + ctx_off]              # (W, L, n_ch)
        Y = data[anchors[:, None] + hz]                    # (W, n_horizons, n_ch)
        return Xc, Y, self.case_ids[anchors], self.step_idx[anchors]

    @classmethod
    def load(cls, out_dir: str | Path, split: str) -> VirtualSensorDataset:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / METADATA_FILE).read_text())
        with np.load(out_dir / f"{split}.npz", allow_pickle=False) as npz:
            return cls(
                X=npz["X"], case_ids=npz["case_ids"], step_idx=npz["step_idx"],
                window_anchors=npz["window_anchors"],
                channels=list(meta["channels"]),
                context_length=meta["context_length"], horizons=list(meta["horizons"]),
                split=split, scaler=meta["scaler"], static_context=meta["static_context"],
                provenance=meta["provenance"],
            )


# --------------------------------------------------------------------- #
def _code_hash() -> str:
    h = hashlib.sha256()
    for p in (_THIS, Path(_slice_reader.__file__)):
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:16]


def _source_manifest(registry: CaseRegistry, slice_ids, case_ids) -> str:
    entries: list = []
    for cid in sorted(case_ids):
        try:
            d = registry[cid]
        except KeyError:
            entries.append([cid, "missing"])
            continue
        for sid in sorted(slice_ids):
            sdir = None if d.slice_root is None else d.slice_root / sid
            for name in (*(f"snapshots_{f}.npy" for f in _FIELDS), "metadata.json", "grid.npz"):
                p = None if sdir is None else sdir / name
                if p is not None and p.exists():
                    st = p.stat()
                    entries.append([f"{cid}/{sid}/{name}", st.st_size, st.st_mtime_ns])
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint(probes: dict, channel_cfg: dict, task_cfg: dict, splits: dict,
                 manifest: str) -> str:
    payload = json.dumps(
        {"probes": probes, "channels": channel_cfg, "task": task_cfg, "splits": splits,
         "code_hash": _code_hash(), "source_manifest": manifest},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _channel_names(probes: dict, fields, aggregates) -> list[str]:
    return [f"{probe}.{f}_{agg}"
            for probe in probes for f in fields for agg in aggregates]


def _gather_case(registry: CaseRegistry, cid: str, probes: dict, fields, aggregates,
                 lo: int, hi: int):
    """(channel matrix (hi-lo, n_ch), static-context dict) for one case."""
    per_probe, y_loc = [], {}
    Pb = Thw = Tcw = None
    for probe, spec in probes.items():
        sc = registry[cid].load_slice(spec["slice_id"], fields=fields)
        fl = {f: np.asarray(sc.snapshots[f])[lo:hi] for f in fields}
        mat, _ = compute_sensor_channels(fl, fields=fields, aggregates=aggregates)
        per_probe.append(mat)
        y_loc[probe] = float(sc.metadata.y_loc)
        Pb, Thw, Tcw = sc.metadata.Pb_Pc, sc.metadata.Thw_Tc, sc.metadata.Tcw_Tc
    ctx = {"Pb_Pc": float(Pb), "Thw_Tc": float(Thw), "Tcw_Tc": float(Tcw), "y_loc": y_loc}
    return np.hstack(per_probe).astype(np.float32), ctx


def _build_split(registry, spec, probes, fields, aggregates, context_length, horizons):
    lo, hi = spec["steps"]
    blocks, cids, steps, anchors, ctx = [], [], [], [], {}
    row0 = 0
    hmax = max(horizons)
    for cid in spec["cases"]:
        mat, cctx = _gather_case(registry, cid, probes, fields, aggregates, lo, hi)
        n = len(mat)
        blocks.append(mat)
        cids.append(np.full(n, cid))
        steps.append(np.arange(lo, lo + n))
        ctx[cid] = cctx
        # anchors valid entirely within this case's block
        first = row0 + context_length - 1
        last = row0 + n - hmax - 1
        if last >= first:
            anchors.append(np.arange(first, last + 1))
        row0 += n
    X = np.concatenate(blocks) if blocks else np.empty((0, 0), np.float32)
    return {
        "X": X,
        "case_ids": np.concatenate(cids) if cids else np.empty(0, "<U8"),
        "step_idx": np.concatenate(steps) if steps else np.empty(0, int),
        "window_anchors": (np.concatenate(anchors) if anchors else np.empty(0, int)).astype(int),
        "static_context": ctx,
    }


def build_virtual_sensor_dataset(
    registry: CaseRegistry,
    config: dict,
    splits: dict,
    *,
    out_dir: str | Path,
    rebuild: bool = False,
) -> dict[str, VirtualSensorDataset]:
    """Build (or load from cache) the three splits.

    `config` is the parsed `virtual_sensors_v1.yaml` (needs `probes`,
    `channels`, `task`); `splits` is the normalised form from
    `normalize_vs_split_config`.
    """
    out_dir = Path(out_dir)
    probes = dict(config["probes"])
    ch_cfg = config["channels"]
    fields = tuple(ch_cfg.get("fields", _FIELDS))
    aggregates = tuple(ch_cfg.get("aggregates", {}).keys()) or _AGGREGATES
    task = config["task"]
    context_length = int(task["context_length"])
    horizons = [int(h) for h in task["horizons"]]
    channels = _channel_names(probes, fields, aggregates)

    all_cases = sorted({c for s in splits.values() for c in s["cases"]})
    slice_ids = sorted({p["slice_id"] for p in probes.values()})
    manifest = _source_manifest(registry, slice_ids, all_cases)
    fp = _fingerprint(probes, ch_cfg, task, splits, manifest)

    meta_path = out_dir / METADATA_FILE
    if not rebuild and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached.get("provenance", {}).get("fingerprint") == fp and all(
            (out_dir / f"{s}.npz").exists() for s in _SPLITS
        ):
            return {s: VirtualSensorDataset.load(out_dir, s) for s in _SPLITS}

    raw = {
        s: _build_split(registry, splits[s], probes, fields, aggregates,
                        context_length, horizons)
        for s in _SPLITS
    }
    train_X = raw["train"]["X"]
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    scaler = {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}

    static_context: dict = {}
    for s in _SPLITS:
        static_context.update(raw[s]["static_context"])

    out_dir.mkdir(parents=True, exist_ok=True)
    for s in _SPLITS:
        np.savez_compressed(
            out_dir / f"{s}.npz",
            X=raw[s]["X"], case_ids=raw[s]["case_ids"],
            step_idx=raw[s]["step_idx"], window_anchors=raw[s]["window_anchors"],
        )
    provenance = {
        "fingerprint": fp,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "data_root": str(registry.data_root),
        "n_steps": {s: len(raw[s]["X"]) for s in _SPLITS},
        "n_windows": {s: len(raw[s]["window_anchors"]) for s in _SPLITS},
    }
    meta_path.write_text(json.dumps({
        "name": config.get("name", "virtual_sensors"),
        "probes": probes, "channels": channels,
        "fields": list(fields), "aggregates": list(aggregates),
        "context_length": context_length, "horizons": horizons,
        "scaler": scaler, "static_context": static_context,
        "splits": splits, "provenance": provenance,
    }, indent=2))
    return {s: VirtualSensorDataset.load(out_dir, s) for s in _SPLITS}

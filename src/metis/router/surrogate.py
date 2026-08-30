"""Wraps pub5_neural_operators' frozen `unet_raw_ood` checkpoint (task A2,
slice `xy_slice_1`, trained on case01-09, blind-evaluated on case10/
case15) as the router's `surrogate_infer()`. See
docs/agent_implementation_plan.md milestone M1.

This is the most-documented checkpoint from the Pub 5 OOD campaign
(FINDINGS §5.13-5.14): U-Net is the "safe default for physical fidelity",
and `campaignB` trained it on the *full* case01-09 grid (not a
leave-one-out split), so it's the right frozen checkpoint for a router
that must answer queries against the whole training envelope rather than
8/9 of it.

Requires the `router` extra (`pip install -e ".[router]"`) plus a
separate editable install of `neuralop_bench` from the sibling
`pub5_neural_operators` checkout — see pyproject.toml.

The checkpoint lives in that sibling project (not under version control
here). Its location is resolved, first hit wins:

    1. $METIS_ROUTER_CHECKPOINT           (full path to best.pt)
    2. $METIS_PUB5_ROOT / runs/.../best.pt (the pub5 checkout root)
    3. a sibling `pub5_neural_operators/` next to the metis repo

so nothing in this file is tied to one machine's home directory.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from neuralop_bench.data import FIELDS, TranscriticalSliceDataset
from neuralop_bench.models import build_model

_CHECKPOINT_RELPATH = Path("runs/campaignB/A2/xy_slice_1/unet_raw_ood/best.pt")
_DEFAULT_PUB5_ROOT = Path(__file__).resolve().parents[4] / "pub5_neural_operators"
PUB5_ROOT = Path(os.environ.get("METIS_PUB5_ROOT", _DEFAULT_PUB5_ROOT))
CHECKPOINT_PATH = Path(
    os.environ.get("METIS_ROUTER_CHECKPOINT", PUB5_ROOT / _CHECKPOINT_RELPATH)
)
SLICE_ID = "xy_slice_1"
TRAIN_CASES = tuple(range(1, 10))


@lru_cache(maxsize=1)
def reference_dataset() -> TranscriticalSliceDataset:
    """The exact training-case dataset the checkpoint was fit on — gives
    the fixed coordinate grid and the `a2_stats` normalization constants
    (mean/std of log10 converged RMS over case01-09). Neither is stored
    in the checkpoint file itself, so this must be reconstructed
    deterministically from the same cases/slice/task/conditioning used
    at train time (see `pub5_neural_operators/scripts/run_campaign.py`).
    """
    return TranscriticalSliceDataset(
        slice_id=SLICE_ID, cases=TRAIN_CASES, task="A2", conditioning="raw", split="train",
    )


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Load the frozen U-Net checkpoint. Architecture hyperparameters
    (`periodic=(False, True)` for an XY wall-bounded-in-y slice, default
    width/depth) aren't stored alongside the checkpoint either — they're
    reconstructed from `run_campaign.py::_make_model`'s defaults for a
    standard (non-ablation) campaign run."""
    model = build_model("unet", in_channels=2, out_channels=3, mu_dim=2, periodic=(False, True))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def surrogate_infer(case_params: dict) -> dict:
    """Run the frozen surrogate for `case_params = {"Pb_Pc": .., "Thw_Tc": ..}`.

    `Tcw_Tc` is not part of the surrogate's own conditioning (only
    `Pb_Pc`/`Thw_Tc` under 'raw' conditioning) — see
    `metis.router.confidence` for why it still matters for trust even
    though the model never sees it.

    Returns physical-space RMS field predictions (u', T', cp'), each
    (H, W), plus their spatial means as a compact summary.
    """
    ds = reference_dataset()
    model = load_model()

    mu = np.array([case_params["Pb_Pc"], case_params["Thw_Tc"]], dtype=np.float32)
    x = torch.from_numpy(ds.coords)[None]
    mu_t = torch.from_numpy(mu)[None]

    with torch.no_grad():
        pred_norm = model(x, mu_t)[0].numpy()
    pred_phys = ds.a2_unnormalize(pred_norm)  # (3, H, W) physical RMS [u, T, cp]

    fields = {name: pred_phys[i] for i, name in enumerate(FIELDS)}
    return {
        "fields": fields,
        "field_means": {name: float(arr.mean()) for name, arr in fields.items()},
    }

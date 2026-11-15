"""Standard physics layer: wall-normal profiles and bulk dimensionless
quantities for a single DNS snapshot.

The formulas here are ported unchanged from the closed-form definitions
already validated against NIST/DNS in
pub4_grassmann_rom/scripts/compute_case_setup_table.py — this module makes
them reusable across cases without reimplementing the physics, operating on
`DNSCase` instead of a raw h5py handle so downstream code never has to know
the RHEA file layout.

Grid convention (RHEA): datasets are (Nz+2, Ny+2, Nx+2) — [z, y, x] axis
order — with one ghost layer per side; y varies along axis 1, and x/z
(axes 2 and 0) are the homogeneous directions averaged out for a
wall-normal profile. Ghosts are wall-mirror cells, so wall values are the
average of the ghost and first interior cell, and wall gradients are
centred at the wall face: (q[1]-q[0])/(y[1]-y[0]).

Bulk velocity is mass-flux-weighted, U_b = <avg_rhou> / <avg_rho>, and bulk
temperature is density-weighted via avg_rhoT, matching the campaign's
constant-mass-flow driving.
"""
from __future__ import annotations

import numpy as np

from metis.data.ingestion.hdf5_reader import DNSCase

DELTA = 175e-6  # channel half-height [m], fixed for the whole DNS series

BULK_FIELDS = (
    "avg_rho", "avg_u", "avg_rhou", "avg_rhoT", "avg_T", "avg_P",
    "avg_mu", "avg_kappa", "avg_c_p", "avg_sos",
)


def wall_normal_profiles(case: DNSCase, field_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """x,z-average the requested fields over interior cells -> y profiles.

    Returns 1D arrays on the full y axis, including the wall-ghost cells
    (only x/z ghosts are stripped), keyed by field name.
    """
    profiles = {}
    for name in field_names:
        q = case.fields[name][1:-1, :, 1:-1]
        profiles[name] = q.mean(axis=(0, 2))
    return profiles


def wall_quantities(profiles: dict[str, np.ndarray], y: np.ndarray, side: str) -> dict[str, float]:
    """Wall value (ghost/interior average) and wall velocity gradient at
    the cold (side='cw', y=0) or hot (side='hw', y=H) wall.

    Requires 'avg_T', 'avg_rho', 'avg_mu', 'avg_u' in `profiles`.
    """
    j0, j1 = (0, 1) if side == "cw" else (-1, -2)
    wall = lambda name: 0.5 * (profiles[name][j0] + profiles[name][j1])
    dy = y[j1] - y[j0]
    dudy = (profiles["avg_u"][j1] - profiles["avg_u"][j0]) / dy
    return {
        "T_w": wall("avg_T"),
        "rho_w": wall("avg_rho"),
        "mu_w": wall("avg_mu"),
        "dudy_w": abs(dudy),
    }


def bulk_quantities(profiles: dict[str, np.ndarray]) -> dict[str, float]:
    """Volume-averaged bulk state and dimensionless groups over interior
    cells (uniform y spacing assumed within the interior).

    Requires the fields in `BULK_FIELDS` to be present in `profiles`.
    """
    inner = slice(1, -1)
    vol = lambda name: float(profiles[name][inner].mean())

    rho_b = vol("avg_rho")
    U_b = vol("avg_rhou") / rho_b
    T_b = vol("avg_rhoT") / rho_b
    mu_b = vol("avg_mu")
    kappa_b = vol("avg_kappa")
    cp_b = vol("avg_c_p")
    c_b = vol("avg_sos")

    return {
        "rho_b": rho_b, "U_b": U_b, "T_b": T_b, "P_b": vol("avg_P"),
        "mu_b": mu_b, "kappa_b": kappa_b, "cp_b": cp_b, "c_b": c_b,
        "Re_b": rho_b * U_b * DELTA / mu_b,
        "Pr_b": mu_b * cp_b / kappa_b,
        "Ec_b": U_b ** 2 / (cp_b * T_b),
        "Br_b": mu_b * U_b ** 2 / (kappa_b * T_b),
        "Ma_b": U_b / c_b,
    }


def friction_reynolds(profiles: dict[str, np.ndarray], y: np.ndarray, side: str) -> dict[str, float]:
    """Wall temperature, friction velocity, and friction Reynolds number
    at the given wall ('cw' or 'hw')."""
    w = wall_quantities(profiles, y, side)
    u_tau = float(np.sqrt(w["mu_w"] * w["dudy_w"] / w["rho_w"]))
    return {"T_w": w["T_w"], "u_tau": u_tau, "Re_tau": w["rho_w"] * u_tau * DELTA / w["mu_w"]}


def case_physics_summary(case: DNSCase) -> dict:
    """Bulk state, dimensionless groups, and both wall friction Reynolds
    numbers for one DNS snapshot — the 'bulk dimensionless groups' feature
    block from research_protocol.md's target-variable list."""
    profiles = wall_normal_profiles(case, BULK_FIELDS)
    y = case.coordinates["y"]

    summary = {"case_id": case.metadata.case_id, **bulk_quantities(profiles)}
    for side in ("cw", "hw"):
        fr = friction_reynolds(profiles, y, side)
        summary[f"T_{side}"] = fr["T_w"]
        summary[f"u_tau_{side}"] = fr["u_tau"]
        summary[f"Re_tau_{side}"] = fr["Re_tau"]
    return summary

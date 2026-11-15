"""I2-B / B6 — evaluate the trained conv-AEs against the PCA baseline.

Reloads every checkpoint listed in ``results/i2b_training.json`` and runs
the protocol §5.1-5.5 battery on the **validation** and **OOD** splits of
the frozen primary slice dataset:

  5.1 reconstruction  — relative_l2 / rmse / mae / nrmse / r2 (standardised)
  5.2 robustness      — seed std of every 5.1 metric, latent_stability
                        (Procrustes across seeds), OOD degradation ratio
  5.3 physical        — mean/RMS profile agreement, x- & z-wavenumber
                        spectrum agreement, POD energy-fraction agreement
  5.4 latent          — max |corr| of any latent axis vs Pb_Pc, Thw_Tc,
                        per-snapshot RMS and spectral-peak wavenumber

PCA (== snapshot POD) is refit once at k=32 and truncated per k. Writes
``results/i2b_evaluation.json``; the accept/stop `assess_model` gate is
B7, which consumes that file.

    python scripts/i2b_evaluate.py --data-root /path/to/dns_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from metis.config import add_data_root_args, resolve_data_root
from metis.data.datasets import SliceDataset
from metis.data.registry import CaseRegistry
from metis.evaluation.metrics import pointwise_metrics
from metis.evaluation.physical import (
    pod_energy_agreement,
    profile_agreement,
    spectrum_agreement,
)
from metis.evaluation.representation import (
    latent_physical_correlation,
    latent_stability,
)
from metis.features.regime import case_grid_labels
from metis.features.spectra import _one_sided_psd
from metis.models import PCARepresentation

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results" / "i2b_training.json"
OUT = REPO / "results" / "i2b_evaluation.json"
_STAB_SUBSET = 300          # snapshots used for the Procrustes latent-stability check
_POD_K = 5                  # leading POD modes compared in pod_energy_agreement


# --------------------------------------------------------------------- #
def _energy_fractions(X_flat: np.ndarray, k: int = 32) -> np.ndarray:
    """Leading POD energy fractions of a centred snapshot ensemble."""
    Xc = np.asarray(X_flat, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return ((sv**2) / np.sum(sv**2))[:k]


def _snapshot_peak_wavenumber(X: np.ndarray) -> np.ndarray:
    """Premultiplied-spectrum peak wavenumber along x, per snapshot."""
    peaks = np.empty(len(X))
    for i, snap in enumerate(X):                      # snap (NX, NZ) -> PSD along NX
        kk, e = _one_sided_psd(snap.T, dx=1.0)
        peaks[i] = kk[np.argmax(kk * e)]
    return peaks


def _recon_block(true_s: np.ndarray, pred_s: np.ndarray, ef_true: np.ndarray) -> dict:
    """5.1 + 5.3 on one (model, split), all in standardised space.
    `ef_true` is the true ensemble's leading POD energy fractions
    (precomputed once per split — it does not depend on the model)."""
    out = dict(pointwise_metrics(true_s, pred_s))
    out["reconstructed_variance"] = 1.0 - float(
        np.mean((true_s - pred_s) ** 2) / np.var(true_s)
    )
    for ax, tag in ((1, "x"), (2, "z")):
        prof = profile_agreement(true_s, pred_s, axis=ax)
        out[f"mean_profile_rel_l2_{tag}"] = prof["mean_profile_rel_l2"]
        out[f"rms_profile_rel_l2_{tag}"] = prof["rms_profile_rel_l2"]
        spec = spectrum_agreement(true_s, pred_s, axis=ax, dx=1.0)
        out[f"spectrum_rel_l2_{tag}"] = spec["spectrum_rel_l2"]
        out[f"spectrum_log_corr_{tag}"] = spec["spectrum_log_corr"]
        out[f"spectrum_peak_rel_shift_{tag}"] = spec["peak_wavenumber_rel_shift"]
    ef_p = _energy_fractions(pred_s.reshape(len(pred_s), -1), _POD_K)
    out["energy_fraction_l1"] = pod_energy_agreement(ef_true, ef_p, k=_POD_K)["energy_fraction_l1"]
    return out


def _phys_vectors(ds: SliceDataset, labels: dict) -> dict[str, np.ndarray]:
    """Per-snapshot physical variables for the latent-correlation check.
    Constant vectors (e.g. Pb_Pc when both OOD cases share a pressure) are
    dropped — a correlation against them is undefined."""
    cids = np.asarray(ds.case_ids, dtype=str)
    raw = {
        "Pb_Pc": np.array([labels[c].Pb_Pc for c in cids], dtype=float),
        "Thw_Tc": np.array([labels[c].Thw_Tc for c in cids], dtype=float),
        "snapshot_rms": ds.X.reshape(len(ds), -1).std(axis=1),
        "peak_wavenumber": _snapshot_peak_wavenumber(ds.X),
    }
    return {name: v for name, v in raw.items() if np.std(v) > 0}


def _lpc(Z: np.ndarray, vectors: dict) -> dict:
    """latent_physical_correlation with the constant-column divide warnings
    (a dead latent axis -> NaN, handled by nanmax) silenced."""
    import warnings

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return latent_physical_correlation(Z, vectors)


def _agg(per_seed: list[dict], key_path) -> tuple[dict, dict]:
    """mean / std over seeds of every leaf under `key_path(seed_dict)`."""
    leaves = [key_path(s) for s in per_seed]
    keys = leaves[0].keys()
    mean = {k: float(np.mean([lf[k] for lf in leaves])) for k in keys}
    std = {k: float(np.std([lf[k] for lf in leaves])) for k in keys}
    return mean, std


# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_args(p)
    p.add_argument("--manifest", default=MANIFEST, type=Path)
    p.add_argument("--output", default=OUT, type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    import torch

    from metis.models.conv_autoencoder import _make_conv_net

    man = json.loads(args.manifest.read_text())
    dataset = REPO / man["dataset"] if not Path(man["dataset"]).is_absolute() else Path(man["dataset"])
    hw = tuple(man["input_hw"])
    channels = tuple(man["channels"])
    activation = man["activation"]
    grid = man["latent_dims"]
    seeds = man["seeds"]
    ckpt_by = {(r["latent_dim"], r["seed"]): REPO / r["checkpoint"] for r in man["runs"]}
    headline = 32

    val = SliceDataset.load(dataset, "val")
    ood = SliceDataset.load(dataset, "ood")
    Xva_s = val.standardize()
    Xoo_s = ood.standardize()
    Xtr_s = SliceDataset.load(dataset, "train").standardize()
    ntr, nva, noo = len(Xtr_s), len(Xva_s), len(Xoo_s)
    tr_flat = Xtr_s.reshape(ntr, -1)

    registry = CaseRegistry(resolve_data_root(cli_value=args.data_root, config_path=args.config))
    case_ids = sorted(set(map(str, val.case_ids)) | set(map(str, ood.case_ids)))
    labels = case_grid_labels(registry, tuple(case_ids))
    phys = {"val": _phys_vectors(val, labels), "ood": _phys_vectors(ood, labels)}

    print(f"val {Xva_s.shape}  ood {Xoo_s.shape}  device={args.device}")

    # true POD energy fractions per split — model-independent, compute once
    ef_true = {
        "val": _energy_fractions(Xva_s.reshape(nva, -1), _POD_K),
        "ood": _energy_fractions(Xoo_s.reshape(noo, -1), _POD_K),
    }

    # --- PCA baseline: one SVD at k=32, truncate per k -----------------
    pca32 = PCARepresentation(latent_dim=max(grid)).fit(tr_flat)

    def pca_recon(Xs_flat, k):
        comp = pca32.components_[:k]
        Z = (Xs_flat - pca32.mean_) @ comp.T
        return Z, (Z @ comp + pca32.mean_)

    results: dict = {}
    for k in grid:
        Zv_p, Xhv_p = pca_recon(Xva_s.reshape(nva, -1), k)
        Zo_p, Xho_p = pca_recon(Xoo_s.reshape(noo, -1), k)
        pca_val = _recon_block(Xva_s, Xhv_p.reshape(nva, *hw), ef_true["val"])
        pca_ood = _recon_block(Xoo_s, Xho_p.reshape(noo, *hw), ef_true["ood"])
        pca_entry = {
            "val": pca_val, "ood": pca_ood,
            "ood_degradation_ratio": pca_ood["relative_l2"] / pca_val["relative_l2"],
            "latent_physical_correlation": {
                "val": _lpc(Zv_p, phys["val"]),
                "ood": _lpc(Zo_p, phys["ood"]),
            },
        }

        ae_seed_rows = []
        Zv_seeds = []
        for seed in seeds:
            net = _make_conv_net(hw, k, channels, activation, seed=0).to(args.device)
            state = torch.load(ckpt_by[(k, seed)], map_location=args.device)["model_state"]
            net.load_state_dict(state)
            net.eval()
            with torch.no_grad():
                tv = torch.as_tensor(Xva_s, dtype=torch.float32, device=args.device)
                to = torch.as_tensor(Xoo_s, dtype=torch.float32, device=args.device)
                Zv = net.encode(tv).cpu().numpy()
                Zo = net.encode(to).cpu().numpy()
                Xhv = net(tv).cpu().numpy()
                Xho = net(to).cpu().numpy()
            v_block = _recon_block(Xva_s, Xhv, ef_true["val"])
            o_block = _recon_block(Xoo_s, Xho, ef_true["ood"])
            ae_seed_rows.append({
                "seed": seed, "val": v_block, "ood": o_block,
                "ood_degradation_ratio": o_block["relative_l2"] / v_block["relative_l2"],
                "latent_physical_correlation": {
                    "val": _lpc(Zv, phys["val"]),
                    "ood": _lpc(Zo, phys["ood"]),
                },
            })
            Zv_seeds.append(Zv[:_STAB_SUBSET])

        val_mean, val_std = _agg(ae_seed_rows, lambda s: s["val"])
        ood_mean, ood_std = _agg(ae_seed_rows, lambda s: s["ood"])
        degr = [s["ood_degradation_ratio"] for s in ae_seed_rows]
        ae_entry = {
            "per_seed": ae_seed_rows,
            "val_mean": val_mean, "val_std": val_std,
            "ood_mean": ood_mean, "ood_std": ood_std,
            "ood_degradation_ratio_mean": float(np.mean(degr)),
            "ood_degradation_ratio_std": float(np.std(degr)),
            "latent_stability": latent_stability(Zv_seeds),
        }
        results[str(k)] = {
            "compression_ratio": (hw[0] * hw[1]) / k,
            "pca": pca_entry, "ae": ae_entry,
        }
        print(f"  k={k:>2}  PCA val relL2 {pca_val['relative_l2']:.3f} / OOD {pca_ood['relative_l2']:.3f}"
              f"   AE val {val_mean['relative_l2']:.3f}±{val_std['relative_l2']:.3f}"
              f" / OOD {ood_mean['relative_l2']:.3f}±{ood_std['relative_l2']:.3f}"
              f"   latent_stab max_relL2 {ae_entry['latent_stability']['max_rel_l2']:.3f}")

    payload = {
        "manifest": str(args.manifest.relative_to(REPO)),
        "dataset": man["dataset"],
        "headline_latent_dim": headline,
        "n": {"train": ntr, "val": nva, "ood": noo},
        "latent_dims": grid, "seeds": seeds,
        "physical_space": "standardised (global train mean/std)",
        "notes": [
            "mean_profile_rel_l2 is noisy: a fluctuation field has ~zero mean profile.",
            "spectrum dx=1.0 (index wavenumbers); relative comparison only.",
        ],
        "per_latent_dim": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    h = results[str(headline)]
    print(f"\nheadline k={headline}:")
    print(f"  reconstruction  PCA val relL2 {h['pca']['val']['relative_l2']:.4f}  "
          f"AE {h['ae']['val_mean']['relative_l2']:.4f} ± {h['ae']['val_std']['relative_l2']:.4f}")
    print(f"  OOD degradation PCA {h['pca']['ood_degradation_ratio']:.3f}  "
          f"AE {h['ae']['ood_degradation_ratio_mean']:.3f} ± {h['ae']['ood_degradation_ratio_std']:.3f}")
    print(f"  RMS-profile relL2 (x)  PCA {h['pca']['val']['rms_profile_rel_l2_x']:.3f}  "
          f"AE {h['ae']['val_mean']['rms_profile_rel_l2_x']:.3f}")
    print(f"  spectrum relL2 (x)     PCA {h['pca']['val']['spectrum_rel_l2_x']:.3f}  "
          f"AE {h['ae']['val_mean']['spectrum_rel_l2_x']:.3f}")
    print(f"  latent_stability max relL2 {h['ae']['latent_stability']['max_rel_l2']:.3f}")
    print(f"\nWrote {args.output}  —  next: B7 assess_model accept/stop gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())

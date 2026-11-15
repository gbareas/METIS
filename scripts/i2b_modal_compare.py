"""I2-B / B8 — modal & physical-statistic comparison, PCA vs conv-AE.

The B7 gate already says stop; this milestone documents *why* the linear
basis wins. Reloads the frozen slice dataset, PCA at the headline k, and
the best conv-AE seed at the same k, then compares the reconstructed
ensembles on:

  * the training POD singular-value spectrum (the linear-rank story);
  * wall-parallel RMS profiles along x and z (val and OOD);
  * 1-D wavenumber spectra E(k) along x and z (val and OOD);
  * leading POD energy-fraction spectra of the true vs reconstructed
    ensembles.

Writes ``results/i2b_modal_compare.json`` (curve data + scalar summary)
and, with the ``report`` extra, PNGs under
``reports/i2b-representation/figures/``. FINDINGS §7 B8 cites it.

    python scripts/i2b_modal_compare.py --data-root /path/to/dns_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from metis.data.datasets import SliceDataset
from metis.evaluation.metrics import relative_l2
from metis.features.spectra import _one_sided_psd
from metis.models import PCARepresentation

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "results" / "i2b_evaluation.json"
MANIFEST = REPO / "results" / "i2b_training.json"
OUT = REPO / "results" / "i2b_modal_compare.json"
REPORT_DIR = REPO / "reports" / "i2b-representation"
_POD_SHOW = 20          # leading POD modes plotted / tabulated


def _ensemble_spectrum(X: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """E(k) along `axis` of an (n, NX, NZ) ensemble, averaged over the rest."""
    return _one_sided_psd(np.moveaxis(X, axis, -1), dx=1.0)


def _rms_profile(X: np.ndarray, keep_axis: int) -> np.ndarray:
    moved = np.moveaxis(X, keep_axis, 0)
    return moved.reshape(moved.shape[0], -1).std(axis=1)


def _energy_fractions(X_flat: np.ndarray, k: int) -> np.ndarray:
    Xc = np.asarray(X_flat, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    return ((sv**2) / np.sum(sv**2))[:k]


def _best_seed(ev: dict, k: str) -> int:
    rows = ev["per_latent_dim"][k]["ae"]["per_seed"]
    return min(rows, key=lambda r: r["val"]["relative_l2"])["seed"]


def _load_ae_net(ckpt: Path, hw, k, channels, activation, device):
    import torch

    from metis.models.conv_autoencoder import _make_conv_net

    net = _make_conv_net(hw, k, channels, activation, seed=0).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    net.eval()
    return net


def _curves(true_s, pca_s, ae_s, hw) -> dict:
    n = len(true_s)
    out = {"n": n}
    for ax, tag in ((1, "x"), (2, "z")):
        kk, e_t = _ensemble_spectrum(true_s, ax)
        _, e_p = _ensemble_spectrum(pca_s, ax)
        _, e_a = _ensemble_spectrum(ae_s, ax)
        out[f"spectrum_{tag}"] = {
            "k": kk.tolist(), "true": e_t.tolist(), "pca": e_p.tolist(), "ae": e_a.tolist(),
            "pca_rel_l2": relative_l2(e_t, e_p), "ae_rel_l2": relative_l2(e_t, e_a),
        }
        r_t = _rms_profile(true_s, ax)
        out[f"rms_profile_{tag}"] = {
            "true": r_t.tolist(),
            "pca": _rms_profile(pca_s, ax).tolist(),
            "ae": _rms_profile(ae_s, ax).tolist(),
            "pca_rel_l2": relative_l2(r_t, _rms_profile(pca_s, ax)),
            "ae_rel_l2": relative_l2(r_t, _rms_profile(ae_s, ax)),
        }
    ef_t = _energy_fractions(true_s.reshape(n, -1), _POD_SHOW)
    out["pod_energy_fraction"] = {
        "true": ef_t.tolist(),
        "pca": _energy_fractions(pca_s.reshape(n, -1), _POD_SHOW).tolist(),
        "ae": _energy_fractions(ae_s.reshape(n, -1), _POD_SHOW).tolist(),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluation", default=EVAL, type=Path)
    p.add_argument("--manifest", default=MANIFEST, type=Path)
    p.add_argument("--output", default=OUT, type=Path)
    p.add_argument("--report-dir", default=REPORT_DIR, type=Path)
    p.add_argument("--seed", type=int, default=None, help="AE seed to use (default: best val relL2)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-figures", dest="figures", action="store_false")
    args = p.parse_args(argv)

    import torch

    ev = json.loads(args.evaluation.read_text())
    man = json.loads(args.manifest.read_text())
    k = ev["headline_latent_dim"]
    ks = str(k)
    seed = args.seed if args.seed is not None else _best_seed(ev, ks)
    hw = tuple(man["input_hw"])
    channels = tuple(man["channels"])
    activation = man["activation"]
    dataset = REPO / man["dataset"]
    ckpt = REPO / next(r["checkpoint"] for r in man["runs"]
                       if r["latent_dim"] == k and r["seed"] == seed)

    val = SliceDataset.load(dataset, "val")
    ood = SliceDataset.load(dataset, "ood")
    Xva_s, Xoo_s = val.standardize(), ood.standardize()
    Xtr_s = SliceDataset.load(dataset, "train").standardize()
    ntr = len(Xtr_s)

    # linear-rank story: full training singular spectrum
    Xc = Xtr_s.reshape(ntr, -1).astype(np.float64)
    Xc -= Xc.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    cum = np.cumsum(sv**2) / np.sum(sv**2)

    pca = PCARepresentation(latent_dim=k).fit(Xtr_s.reshape(ntr, -1))

    def recon_pca(Xs):
        flat = Xs.reshape(len(Xs), -1)
        return (pca.inverse_transform(pca.transform(flat))).reshape(Xs.shape)

    net = _load_ae_net(ckpt, hw, k, channels, activation, args.device)

    def recon_ae(Xs):
        with torch.no_grad():
            t = torch.as_tensor(Xs, dtype=torch.float32, device=args.device)
            return net(t).cpu().numpy()

    splits = {}
    for name, Xs in (("val", Xva_s), ("ood", Xoo_s)):
        splits[name] = _curves(Xs, recon_pca(Xs), recon_ae(Xs), hw)

    val_c, ood_c = splits["val"], splits["ood"]
    summary = {
        "headline_latent_dim": k,
        "ae_seed": seed,
        "linear_rank": {
            "reconstructed_variance_at_k": float(cum[k - 1]),
            "modes_for_90pct": int(np.argmax(cum >= 0.90) + 1) if np.any(cum >= 0.90) else None,
            "modes_for_99pct": int(np.argmax(cum >= 0.99) + 1) if np.any(cum >= 0.99) else None,
        },
        "val_spectrum_rel_l2": {"x_pca": val_c["spectrum_x"]["pca_rel_l2"],
                                "x_ae": val_c["spectrum_x"]["ae_rel_l2"],
                                "z_pca": val_c["spectrum_z"]["pca_rel_l2"],
                                "z_ae": val_c["spectrum_z"]["ae_rel_l2"]},
        "ood_spectrum_rel_l2": {"x_pca": ood_c["spectrum_x"]["pca_rel_l2"],
                                "x_ae": ood_c["spectrum_x"]["ae_rel_l2"],
                                "z_pca": ood_c["spectrum_z"]["pca_rel_l2"],
                                "z_ae": ood_c["spectrum_z"]["ae_rel_l2"]},
        "val_rms_profile_rel_l2": {"x_pca": val_c["rms_profile_x"]["pca_rel_l2"],
                                   "x_ae": val_c["rms_profile_x"]["ae_rel_l2"]},
        "ood_rms_profile_rel_l2": {"x_pca": ood_c["rms_profile_x"]["pca_rel_l2"],
                                   "x_ae": ood_c["rms_profile_x"]["ae_rel_l2"]},
    }

    payload = {
        "dataset": man["dataset"],
        "checkpoint": str(ckpt.relative_to(REPO)),
        "training_singular_values_head": sv[:64].tolist(),
        "training_cumulative_energy_head": cum[:64].tolist(),
        "splits": splits,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"headline k={k}  AE seed={seed}  ({ckpt.name})")
    lr = summary["linear_rank"]
    print(f"  training POD: {lr['reconstructed_variance_at_k']*100:.1f}% variance in {k} modes; "
          f"{lr['modes_for_90pct']} modes for 90%, {lr['modes_for_99pct']} for 99%")
    for sp, c in (("val", val_c), ("ood", ood_c)):
        print(f"  [{sp}] x-spectrum relL2   PCA {c['spectrum_x']['pca_rel_l2']:.3f}   "
              f"AE {c['spectrum_x']['ae_rel_l2']:.3f}")
        print(f"  [{sp}] x-RMS-profile relL2 PCA {c['rms_profile_x']['pca_rel_l2']:.3f}   "
              f"AE {c['rms_profile_x']['ae_rel_l2']:.3f}")

    if args.figures:
        _write_figures(args.report_dir, payload, Xva_s, recon_pca(Xva_s), recon_ae(Xva_s))
    print(f"\nWrote {args.output}  —  next: B9 metis-report generator + §7 polish")
    return 0


def _write_figures(report_dir: Path, payload: dict, true_s, pca_s, ae_s) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figures; pip install -e \".[report]\")")
        return

    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cum = np.asarray(payload["training_cumulative_energy_head"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker=".")
    ax.axhline(0.9, ls="--", c="grey"); ax.axhline(0.99, ls=":", c="grey")
    ax.set(xlabel="POD mode", ylabel="cumulative reconstructed variance",
           title="Training POD spectrum (centre-plane u')")
    fig.tight_layout(); fig.savefig(fig_dir / "pod_spectrum.png", dpi=120); plt.close(fig)

    for split in ("val", "ood"):
        c = payload["splits"][split]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, tag in zip(axes, ("x", "z")):
            s = c[f"spectrum_{tag}"]
            ax.loglog(s["k"][1:], s["true"][1:], label="true", lw=2)
            ax.loglog(s["k"][1:], s["pca"][1:], label="PCA", ls="--")
            ax.loglog(s["k"][1:], s["ae"][1:], label="conv-AE", ls=":")
            ax.set(xlabel=f"$k_{tag}$", ylabel="E(k)",
                   title=f"{split} spectrum along {tag}")
            ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"spectrum_{split}.png", dpi=120); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, tag in zip(axes, ("x", "z")):
            r = c[f"rms_profile_{tag}"]
            ax.plot(r["true"], label="true", lw=2)
            ax.plot(r["pca"], label="PCA", ls="--")
            ax.plot(r["ae"], label="conv-AE", ls=":")
            ax.set(xlabel=f"{tag} index", ylabel="RMS", title=f"{split} RMS profile along {tag}")
            ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / f"rms_profile_{split}.png", dpi=120); plt.close(fig)

    ef = payload["splits"]["val"]["pod_energy_fraction"]
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, len(ef["true"]) + 1)
    ax.plot(idx, ef["true"], marker="o", label="true")
    ax.plot(idx, ef["pca"], marker="s", label="PCA recon", ls="--")
    ax.plot(idx, ef["ae"], marker="^", label="conv-AE recon", ls=":")
    ax.set(xlabel="POD mode", ylabel="energy fraction", yscale="log",
           title="val: POD energy spectrum of reconstructed ensembles")
    ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / "pod_energy_fraction_val.png", dpi=120); plt.close(fig)

    rng = np.random.default_rng(0)
    picks = rng.choice(len(true_s), size=3, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, i in enumerate(picks):
        for col, (field, ttl) in enumerate(
            ((true_s[i], "true"), (pca_s[i], "PCA"), (ae_s[i], "conv-AE"))
        ):
            a = axes[row, col]
            a.imshow(field, cmap="RdBu_r", vmin=-3, vmax=3)
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(ttl)
        axes[row, 0].set_ylabel(f"val snap {i}")
    fig.suptitle("Sample reconstructions (standardised u')")
    fig.tight_layout(); fig.savefig(fig_dir / "sample_fields_val.png", dpi=120); plt.close(fig)
    print(f"  figures -> {fig_dir}")


if __name__ == "__main__":
    sys.exit(main())

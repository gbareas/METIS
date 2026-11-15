"""End-to-end pipeline integration test (Stage 3 / §24).

ingest -> validate -> preprocess -> build dataset -> fit a model ->
evaluate -> save/reload artifacts -> report. Runs on synthetic data
(`metis.testing`), so it needs no real DNS. The torch leg
(`Autoencoder` + `Trainer` + MLflow) is skipped when the `ml` extra
isn't installed.
"""
import shutil

import numpy as np
import pytest

from metis.analysis import run_analysis
from metis.data.datasets import build_feature_dataset
from metis.data.preprocessing import StandardScaler
from metis.data.registry import CaseRegistry
from metis.data.validation import validate_case
from metis.evaluation.assessment import assess_model
from metis.evaluation.metrics import pointwise_metrics
from metis.evaluation.physical import physical_field_report
from metis.evaluation.representation import evaluate_representation
from metis.models import PCARepresentation
from metis.reporting.generators import physics_report
from metis.testing import mock_dns, mock_slices

CASES = [f"case{n:02d}" for n in (1, 2, 3, 4, 5, 6)]
RICH_SLICES = ("s3_center", "s5_y_plus_10_cw", "s8_y_plus_10_hw")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("dns")
    for i, cid in enumerate(CASES, start=1):
        raw = root / "raw" / cid
        _h5, meta = mock_dns.generate(
            raw / f"s_{i:05d}.h5", nx=6, ny=10, nz=6, seed=i, case_id=cid,
            Pb_Pc=[1.5, 2.0, 5.0][(i - 1) % 3], Thw_Tc=[1.1, 1.2, 1.4][(i - 1) // 3 % 3],
            Tcw_Tc=0.95,
        )
        proc = root / "processed" / cid
        proc.mkdir(parents=True)
        shutil.copy(meta, proc / "metadata.json")
        meta.unlink()
        for sid in RICH_SLICES:
            mock_slices.generate(root / "processed_slices" / cid / sid, nx=8, nz=8,
                                 n_snapshots=16, seed=i, case_id=cid, slice_id=sid)
    return CaseRegistry(root)


def test_ingest_validate_dataset_model_evaluate_report(registry, tmp_path):
    # -- ingest + validate ------------------------------------------
    assert registry.case_ids() == CASES
    for cid in CASES:
        assert validate_case(registry[cid]).ok

    # -- build a cached feature dataset ----------------------------
    art = tmp_path / "ds"
    ds1 = build_feature_dataset(registry, CASES, "bulk", out_dir=art)
    ds2 = build_feature_dataset(registry, CASES, "bulk", out_dir=art)  # cache hit
    assert ds1.provenance["created_at"] == ds2.provenance["created_at"]
    np.testing.assert_array_equal(ds1.X, ds2.X)
    assert ds1.X.shape == (6, 7)

    # -- preprocess (fit on a train split only) --------------------
    train = slice(0, 4)
    scaler = StandardScaler().fit(ds1.X[train])
    Xs = scaler.transform(ds1.X)
    np.testing.assert_allclose(scaler.mean_, ds1.X[train].mean(axis=0))

    # -- fit a model + evaluate ----------------------------------
    model = PCARepresentation(latent_dim=2).fit(Xs[train])
    Z = model.transform(Xs)
    recon = model.inverse_transform(Z)
    ml = pointwise_metrics(Xs, recon)
    # the mock grid: Pb_Pc cycles 1.5/2.0/5.0, Thw_Tc steps 1.1 then 1.2
    pb = np.array([1.5, 2.0, 5.0, 1.5, 2.0, 5.0])
    thw = np.array([1.1, 1.1, 1.1, 1.2, 1.2, 1.2])
    rep = evaluate_representation(Z, pb, thw)
    assert 0.0 <= ml["relative_l2"]
    assert set(rep) >= {"ari_vs_Pb_Pc", "loco_accuracy_Thw_Tc"}

    # a worse model must not pass the §17 "is it better?" gate
    verdict = assess_model(
        {"rmse": ml["rmse"] * 2, "profile_rel_l2": 0.5},
        {"rmse": ml["rmse"], "profile_rel_l2": 0.1},
        ml_metrics=["rmse"], physical_metrics=["profile_rel_l2"],
    )
    assert not verdict.is_better

    # -- analysis result: save + reload --------------------------
    result = run_analysis(registry, "physics", "case01")
    assert result.validation["ok"]
    saved = result.save(tmp_path / "an")
    from metis.analysis import AnalysisResult

    reloaded = AnalysisResult.load(saved)
    assert reloaded.outputs == result.outputs
    np.testing.assert_array_equal(reloaded.arrays["y"], result.arrays["y"])

    # -- physical-diagnostic agreement of a field vs a perturbation
    field = result.arrays["avg_T"]
    pf = physical_field_report(field, field * 1.05, profile_axis=0)
    assert pf["mean_profile_rel_l2"] == pytest.approx(0.05, rel=1e-6)

    # -- report ------------------------------------------------
    out = physics_report(result).write(tmp_path / "rep")
    assert (out / "summary.md").exists()
    assert (out / "metrics.json").exists()


def test_training_leg_end_to_end(registry, tmp_path):
    pytest.importorskip("torch")
    from metis import tracking
    from metis.models.autoencoder import Autoencoder
    from metis.training import Trainer

    ds = build_feature_dataset(registry, CASES, "rms_profile")
    scaler = StandardScaler().fit(ds.X[:4])
    Xs = scaler.transform(ds.X)

    with tracking.run("pipeline-test", params={"model": "ae"},
                      tracking_dir=tmp_path / "mlruns") as run:
        ae = Autoencoder(latent_dim=2, hidden=(16,), standardize=False)
        ae.fit(Xs[:4], trainer=Trainer(seed=0, max_epochs=400, patience=100,
                                       log_every=100), run=run)
        run.log_metrics({"final_mse": ae.history_["best_train_mse"]})

    assert ae.is_fitted
    assert ae.transform(Xs).shape == (6, 2)
    assert ae.history_["best_train_mse"] < ae.history_["logged_losses"][0]

    # serialise + restore the trained model
    from metis.data.preprocessing import from_dict, to_dict

    clone = from_dict(to_dict(ae))
    np.testing.assert_allclose(clone.transform(Xs), ae.transform(Xs), atol=1e-5)

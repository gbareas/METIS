"""The frozen `regime-v1` benchmark (milestone R7).

Runs the settled regime-discovery evaluation (FINDINGS.md §1-4) as a
reproducible platform acceptance test: per-block ARI/LOCO, plus the MFA
vs. naive combination of the pressure and thermal specialist blocks. The
`checks` on the result encode the frozen conclusions; `assert_passes`
turns any regression into an `AssertionError`.

Feed it block feature matrices from `metis.data.datasets` (or the cached
`tests/data/regime_v1/block_features.npz` fixture for a DNS-free run).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from metis.config import REPO_ROOT, git_commit
from metis.evaluation.regime import combine_blocks_mfa, evaluate_regime_discovery
from metis.features.regime import ALL_CASE_IDS, FEATURE_BLOCK_NAMES, CaseGridLabel

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "benchmarks" / "regime_discovery_v1.yaml"

_METRIC_KEYS = ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc", "loco_accuracy_Pb_Pc", "loco_accuracy_Thw_Tc")


@dataclass(frozen=True)
class BenchmarkCheck:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class BenchmarkResult:
    name: str
    blocks: dict[str, dict]          # block -> evaluate_regime_discovery output
    combined: dict[str, dict]        # "mfa" / "naive" -> evaluate output
    checks: list[BenchmarkCheck]
    provenance: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        head = f"{self.name}: {'PASS' if self.passed else 'FAIL'} " \
               f"({sum(c.passed for c in self.checks)}/{len(self.checks)} checks)"
        return "\n".join([head, *(f"  {c}" for c in self.checks)])

    def assert_passes(self) -> None:
        if not self.passed:
            fails = "\n".join(f"  {c}" for c in self.checks if not c.passed)
            raise AssertionError(f"{self.name} regressed:\n{fails}")


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text())


def _labels_from_arrays(case_ids, Pb_Pc, Thw_Tc) -> dict[str, CaseGridLabel]:
    return {
        str(c): CaseGridLabel(Pb_Pc=float(p), Thw_Tc=float(t))
        for c, p, t in zip(case_ids, Pb_Pc, Thw_Tc)
    }


def run_regime_v1(
    block_matrices: dict[str, np.ndarray],
    labels: dict[str, CaseGridLabel],
    config: dict | None = None,
) -> BenchmarkResult:
    """Evaluate every block plus the MFA / naive combination and score the
    frozen checks. `block_matrices[b]` rows must be in `ALL_CASE_IDS`
    order."""
    config = config or load_config()
    tol = float(config.get("tolerance", 1e-6))
    exp = config["expected"]
    pressure_block = config["diagnostics"]["pressure_block"]
    thermal_block = config["diagnostics"]["thermal_block"]
    combo = list(config["combined_sanity"])

    blocks = {
        b: evaluate_regime_discovery(block_matrices[b], labels)
        for b in FEATURE_BLOCK_NAMES
    }
    naive_X = np.concatenate([block_matrices[b] for b in combo], axis=1)
    mfa_X = combine_blocks_mfa({b: block_matrices[b] for b in combo})
    combined = {
        "naive": evaluate_regime_discovery(naive_X, labels),
        "mfa": evaluate_regime_discovery(mfa_X, labels, standardize_input=False),
    }

    checks: list[BenchmarkCheck] = []

    def _best_block(metric: str) -> str:
        return max(FEATURE_BLOCK_NAMES, key=lambda b: blocks[b][metric])

    for metric, expected_block in (
        ("ari_vs_Pb_Pc", pressure_block),
        ("loco_accuracy_Pb_Pc", pressure_block),
        ("ari_vs_Thw_Tc", thermal_block),
        ("loco_accuracy_Thw_Tc", thermal_block),
    ):
        best = _best_block(metric)
        checks.append(BenchmarkCheck(
            f"best_block__{metric}", best == expected_block,
            f"expected {expected_block}, got {best} "
            f"({', '.join(f'{b}={blocks[b][metric]:.3f}' for b in FEATURE_BLOCK_NAMES)})",
        ))

    # MFA of the two specialists puts both OOD cases nearest Pb_Pc=1.5;
    # naive concatenation puts them nearest 5.0 (H3).
    mfa_ood = combined["mfa"]["ood_nearest_Pb_Pc_centroid"]
    naive_ood = combined["naive"]["ood_nearest_Pb_Pc_centroid"]
    checks.append(BenchmarkCheck(
        "mfa_restores_ood_pressure",
        mfa_ood == {c: float(v) for c, v in exp["mfa_combined"]["ood_nearest_Pb_Pc_centroid"].items()},
        f"MFA OOD nearest-centroid {mfa_ood}",
    ))
    checks.append(BenchmarkCheck(
        "naive_concat_gets_ood_wrong",
        naive_ood == {c: float(v) for c, v in exp["naive_combined"]["ood_nearest_Pb_Pc_centroid"].items()},
        f"naive OOD nearest-centroid {naive_ood} (control: should be the wrong value)",
    ))

    # Deterministic metric values must not drift.
    drift: list[str] = []
    for b in FEATURE_BLOCK_NAMES:
        for k in _METRIC_KEYS:
            got, want = blocks[b][k], exp["blocks"][b][k]
            if abs(got - want) > tol:
                drift.append(f"{b}.{k}: {got:.6f} != {want:.6f}")
    for k in ("ari_vs_Pb_Pc", "ari_vs_Thw_Tc"):
        got, want = combined["mfa"][k], exp["mfa_combined"][k]
        if abs(got - want) > tol:
            drift.append(f"mfa.{k}: {got:.6f} != {want:.6f}")
    checks.append(BenchmarkCheck(
        "no_metric_drift", not drift,
        "all frozen metrics within tolerance" if not drift else "; ".join(drift),
    ))

    return BenchmarkResult(
        name=config["name"],
        blocks=blocks,
        combined=combined,
        checks=checks,
        provenance={
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": git_commit(),
        },
    )


DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "datasets" / "regime-v1"


def run_regime_v1_from_registry(
    registry,
    config: dict | None = None,
    *,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
) -> BenchmarkResult:
    """Build the block feature matrices from a `CaseRegistry` and run the
    benchmark. Feature blocks are cached under
    `artifacts/datasets/regime-v1/<block>/` (pass `cache_dir=None` to
    disable, `rebuild=True` to force). `result.provenance["cache"]`
    records which blocks were reused vs. rebuilt."""
    from metis.data.datasets import FeatureDataset, build_feature_dataset
    from metis.data.datasets.build import _fingerprint

    config = config or load_config()
    case_ids = list(config["cases"]["train"]) + list(config["cases"]["ood"])
    assert case_ids == list(ALL_CASE_IDS), (
        f"benchmark case list {case_ids} != regime ALL_CASE_IDS {list(ALL_CASE_IDS)}"
    )

    cache: dict[str, str] = {}
    block_matrices = {}
    for b in FEATURE_BLOCK_NAMES:
        out_dir = None if cache_dir is None else Path(cache_dir) / b
        reused = (
            out_dir is not None and not rebuild and FeatureDataset.exists_at(out_dir)
            and FeatureDataset.load(out_dir).fingerprint == _fingerprint(b, case_ids, registry)
        )
        ds = build_feature_dataset(registry, case_ids, b, out_dir=out_dir, rebuild=rebuild)
        block_matrices[b] = ds.X
        cache[b] = "reused" if reused else "built"

    from metis.features.regime import case_grid_labels

    labels = case_grid_labels(registry, tuple(case_ids))  # covers train + OOD
    result = run_regime_v1(block_matrices, labels, config)
    result.provenance["cache"] = cache
    return result

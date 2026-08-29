import shutil

import numpy as np
import pytest

from metis.features.regime import (
    FEATURE_BLOCK_NAMES,
    FEATURE_NAMES,
    POD_K,
    RICH_POD_SLICE_IDS,
    block_feature_names,
    build_feature_matrix,
    build_feature_matrix_block,
    build_feature_matrix_rich,
    build_feature_vector,
    build_feature_vector_block,
    build_feature_vector_rich,
    case_grid_labels,
    rich_feature_names,
)
from metis.testing import mock_dns, mock_slices


def _make_mock_case(data_root, case_id, seed, Pb_Pc, Thw_Tc, Tcw_Tc, n_snapshots=20):
    """Lay out one case under data_root/{raw,processed,processed_slices}/
    matching the real group data layout (see metis.features.regime)."""
    raw_dir = data_root / "raw" / case_id
    _h5_path, meta_path = mock_dns.generate(
        raw_dir / f"snap_{seed:07d}.h5",
        nx=6, ny=8, nz=6, seed=seed,
        case_id=case_id, Pb_Pc=Pb_Pc, Thw_Tc=Thw_Tc, Tcw_Tc=Tcw_Tc,
    )
    processed_dir = data_root / "processed" / case_id
    processed_dir.mkdir(parents=True)
    shutil.copy(meta_path, processed_dir / "metadata.json")
    meta_path.unlink()  # keep only the sibling copy, matching the real layout

    for slice_id in RICH_POD_SLICE_IDS:
        mock_slices.generate(
            data_root / "processed_slices" / case_id / slice_id,
            nx=8, nz=8, n_snapshots=n_snapshots, seed=seed,
            case_id=case_id, slice_id=slice_id, Pb_Pc=Pb_Pc, Thw_Tc=Thw_Tc, Tcw_Tc=Tcw_Tc,
        )


@pytest.fixture
def mock_data_root(tmp_path):
    _make_mock_case(tmp_path, "case01", seed=1, Pb_Pc=1.5, Thw_Tc=1.1, Tcw_Tc=0.95)
    _make_mock_case(tmp_path, "case02", seed=2, Pb_Pc=1.5, Thw_Tc=1.2, Tcw_Tc=0.90)
    return tmp_path


def test_build_feature_vector_shape_and_finite(mock_data_root):
    vec = build_feature_vector("case01", mock_data_root)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_pod_energy_fraction_features_are_leading_k(mock_data_root):
    vec = build_feature_vector("case01", mock_data_root)
    pod_fracs = vec[-POD_K:]
    assert np.all(pod_fracs >= 0.0)
    assert np.all(np.diff(pod_fracs) <= 1e-12)  # non-increasing, leading modes


def test_build_feature_matrix_stacks_cases(mock_data_root):
    X = build_feature_matrix(("case01", "case02"), mock_data_root)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert not np.allclose(X[0], X[1])  # different seeds/conditions -> different vectors


def test_case_grid_labels_reads_true_conditions(mock_data_root):
    labels = case_grid_labels(mock_data_root, ("case01", "case02"))
    assert labels["case01"].Pb_Pc == pytest.approx(1.5)
    assert labels["case01"].Thw_Tc == pytest.approx(1.1)
    assert labels["case02"].Thw_Tc == pytest.approx(1.2)


def test_build_feature_vector_raises_if_too_few_pod_snapshots(mock_data_root, tmp_path):
    case_id = "case_short"
    raw_dir = tmp_path / "raw" / case_id
    _, meta_path = mock_dns.generate(raw_dir / "snap_0000001.h5", nx=6, ny=8, nz=6, seed=9, case_id=case_id)
    processed_dir = tmp_path / "processed" / case_id
    processed_dir.mkdir(parents=True)
    shutil.copy(meta_path, processed_dir / "metadata.json")

    mock_slices.generate(
        tmp_path / "processed_slices" / case_id / "s3_center",
        nx=8, nz=8, n_snapshots=POD_K - 1, seed=9, case_id=case_id, slice_id="s3_center",
    )
    with pytest.raises(AssertionError):
        build_feature_vector(case_id, tmp_path)


def test_rich_feature_vector_length_matches_names(mock_data_root):
    names = rich_feature_names("case01", mock_data_root)
    vec = build_feature_vector_rich("case01", mock_data_root)
    assert vec.shape == (len(names),)
    assert np.all(np.isfinite(vec))


def test_rich_feature_names_are_unique(mock_data_root):
    names = rich_feature_names("case01", mock_data_root)
    assert len(names) == len(set(names))


def test_rich_feature_vector_differs_across_cases(mock_data_root):
    vec1 = build_feature_vector_rich("case01", mock_data_root)
    vec2 = build_feature_vector_rich("case02", mock_data_root)
    assert vec1.shape == vec2.shape
    assert not np.allclose(vec1, vec2)


def test_rich_feature_matrix_stacks_cases_with_matching_names(mock_data_root):
    X, names = build_feature_matrix_rich(("case01", "case02"), mock_data_root)
    assert X.shape == (2, len(names))


def test_rich_feature_vector_much_longer_than_compact(mock_data_root):
    compact = build_feature_vector("case01", mock_data_root)
    rich = build_feature_vector_rich("case01", mock_data_root)
    assert rich.shape[0] > compact.shape[0]


@pytest.mark.parametrize("block", FEATURE_BLOCK_NAMES)
def test_block_feature_vector_shape_and_finite(mock_data_root, block):
    names = block_feature_names("case01", mock_data_root, block)
    vec = build_feature_vector_block("case01", mock_data_root, block)
    assert vec.shape == (len(names),)
    assert np.all(np.isfinite(vec))


@pytest.mark.parametrize("block", FEATURE_BLOCK_NAMES)
def test_block_feature_matrix_stacks_cases(mock_data_root, block):
    X, names = build_feature_matrix_block(("case01", "case02"), mock_data_root, block)
    assert X.shape == (2, len(names))
    assert not np.allclose(X[0], X[1])


def test_blocks_concatenate_to_the_rich_vector(mock_data_root):
    rich = build_feature_vector_rich("case01", mock_data_root)
    concatenated = np.concatenate(
        [build_feature_vector_block("case01", mock_data_root, b) for b in FEATURE_BLOCK_NAMES]
    )
    np.testing.assert_allclose(concatenated, rich)


def test_unknown_block_raises(mock_data_root):
    with pytest.raises(ValueError, match="unknown feature block"):
        build_feature_vector_block("case01", mock_data_root, "not_a_block")

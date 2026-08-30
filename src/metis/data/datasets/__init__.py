"""Dataset abstraction + artifact caching (milestone R5).

    from metis.data.datasets import build_feature_dataset, FeatureDataset

Level-1 (one feature row per case) only for now; Levels 2-4
(local / temporal / full-field) come later.
"""
from metis.data.datasets.build import FEATURE_SETS, build_feature_dataset
from metis.data.datasets.feature_dataset import FeatureDataset
from metis.data.datasets.slice_dataset import (
    SliceDataset,
    build_slice_dataset,
    normalize_split_config,
)

__all__ = [
    "FEATURE_SETS",
    "FeatureDataset",
    "SliceDataset",
    "build_feature_dataset",
    "build_slice_dataset",
    "normalize_split_config",
]

"""Dataset abstraction + artifact caching (milestone R5).

    from metis.data.datasets import build_feature_dataset, FeatureDataset

Level-1 (one feature row per case) only for now; Levels 2-4
(local / temporal / full-field) come later.
"""
from metis.data.datasets.build import FEATURE_SETS, build_feature_dataset
from metis.data.datasets.feature_dataset import FeatureDataset

__all__ = ["FEATURE_SETS", "FeatureDataset", "build_feature_dataset"]

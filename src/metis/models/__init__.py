"""Models (milestone I4). Only what corresponds to a real experiment:

- `PCARepresentation` — the linear baseline (numpy only).
- `Autoencoder` — small MLP autoencoder for the I2 representation study.
- `ConvAutoencoder` — conv autoencoder for the I2-B slice study.

The torch models need the `ml` extra; import them directly to keep torch
off the base import path.

    from metis.models import PCARepresentation
    from metis.models.autoencoder import Autoencoder
    from metis.models.conv_autoencoder import ConvAutoencoder
"""
from metis.models.base import RepresentationModel
from metis.models.pca import PCARepresentation

__all__ = ["PCARepresentation", "RepresentationModel"]

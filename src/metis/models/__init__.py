"""Models (milestone I4). Only what corresponds to a real experiment:

- `PCARepresentation` — the linear baseline (numpy only).
- `Autoencoder` — small MLP autoencoder for the I2 representation study
  (needs the `ml` extra for torch; import it directly to keep torch off
  the base import path).

    from metis.models import PCARepresentation
    from metis.models.autoencoder import Autoencoder
"""
from metis.models.base import RepresentationModel
from metis.models.pca import PCARepresentation

__all__ = ["PCARepresentation", "RepresentationModel"]

"""Training (milestone I3).

    from metis.training import Trainer

`Trainer` needs torch at call time (`ml` extra); importing it here does
not import torch.
"""
from metis.training.trainer import Trainer

__all__ = ["Trainer"]

"""metis — reusable ML analysis & physical-discovery framework."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metis")
except PackageNotFoundError:  # not installed (e.g. run from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

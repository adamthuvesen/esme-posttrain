"""Post-training tools for SFT, DPO, RLVR prep, and dense-bundle export."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("esme-posttrain")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]

"""Core normalized-table access for FlyWire and Male CNS connectomes."""
import warnings as _warnings

_warnings.warn(
    "the 'cex' package is deprecated: use 'connexplorer' (see docs/migrating_from_cex.md)",
    DeprecationWarning,
    stacklevel=2,
)


from .dataset import ConnectomeDataset, FlyWireDataset, MaleCNSDataset, get_dataset

__all__ = ["ConnectomeDataset", "FlyWireDataset", "MaleCNSDataset", "get_dataset"]

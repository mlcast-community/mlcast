"""Data loading for mlcast.

The public classes are imported lazily (PEP 562) so that importing
:mod:`mlcast.data.source_data.sampling`, as the data-prep CLI commands do,
does not pull in torch and Lightning.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .source_data_datamodule import SourceDataDataModule
    from .source_data_datasets import SourceDataIndexedDataset

__all__ = ["SourceDataDataModule", "SourceDataIndexedDataset"]

_LAZY_ATTRS = {
    "SourceDataDataModule": ".source_data_datamodule",
    "SourceDataIndexedDataset": ".source_data_datasets",
}


def __getattr__(name: str) -> object:
    """Import the public classes on first access."""
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

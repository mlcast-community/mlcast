"""PyTorch Lightning data module for spatio-temporal datasets.

Handles train/val/test splitting and DataLoader creation from an injected
dataset factory.
"""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class SourceDataDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for spatio-temporal datasets.

    Handles train/val/test splitting and DataLoader creation by utilizing
    an injected ``dataset_factory``.

    Parameters
    ----------
    dataset_factory : Callable[..., Dataset]
        A factory function (e.g., ``fdl.Partial``) that returns a Dataset instance.
        It must accept ``time_slice`` and ``augment`` as keyword arguments.
    train_ratio : float, optional
        Fraction of data used for training. Default is ``0.7``.
    val_ratio : float, optional
        Fraction of data used for validation. Default is ``0.15``.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.,
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        dataset_factory: Callable[..., Dataset],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_factory = dataset_factory
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage: str | None = None) -> None:
        """Create train, validation, and test datasets.

        Splits are chronological based on the total number of timesteps in
        the Zarr store, determined by opening it directly before instantiating
        any Dataset objects.
        """
        # We need the total number of timesteps to compute split boundaries.
        # Duck-type the factory to extract zarr_path and storage_options —
        # functools.partial stores kwargs in .keywords, fdl.Partial exposes
        # them as attributes.
        zarr_path = getattr(self.dataset_factory, "zarr_path", None) or self.dataset_factory.keywords["zarr_path"]
        storage_options = getattr(self.dataset_factory, "storage_options", None) or self.dataset_factory.keywords.get(
            "storage_options"
        )
        n = xr.open_zarr(zarr_path, storage_options=storage_options).sizes["time"]

        train_end = int(n * self.train_ratio)
        # Compute val_end independently from train_end rather than from the
        # accumulated sum of ratios, to avoid floating-point truncation errors
        # (e.g. int(240 * (0.5 + 1/3)) = int(199.999...) = 199 instead of 200).
        val_end = train_end + int(n * self.val_ratio)

        self.train_dataset = self.dataset_factory(
            time_slice=slice(0, train_end),
            augment=True,
        )
        self.val_dataset = self.dataset_factory(
            time_slice=slice(train_end, val_end),
            augment=False,
        )
        self.test_dataset = self.dataset_factory(
            time_slice=slice(val_end, n),
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)

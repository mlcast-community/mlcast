"""PyTorch Lightning data module for spatio-temporal datasets.

Handles train/val/test splitting and DataLoader creation from an injected
dataset factory.
"""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from mlcast.data.splits import (
    compute_split_ranges_from_splitting_ratios,
    splitting_uses_fractions,
    splitting_uses_tuple_ranges,
    validate_splits,
)


class SourceDataDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for spatio-temporal datasets.

    Handles train/val/test splitting and DataLoader creation by utilizing
    an injected ``dataset_factory``.

    Parameters
    ----------
    dataset_factory : Callable[..., Dataset]
        A factory function (e.g., ``fdl.Partial``) that returns a Dataset instance.
        It must accept ``subset`` and ``augment`` as keyword arguments.
    splits : dict of {str: dict}
        Nested mapping ``{coord: {split_name: value, ...}, ...}`` describing
        train/val/test subsets. Currently only the ``time`` coordinate is
        supported. Ratio mode uses float fractions, while datetime mode uses
        inclusive ``(start, end)`` ISO 8601 string tuples.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.,
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        dataset_factory: Callable[..., Dataset],
        splits: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_factory = dataset_factory
        self.splits = splits
        self.dataloader_kwargs = dataloader_kwargs
        validate_splits(self.splits)

    def setup(self, stage: str | None = None) -> None:
        """Create train, validation, and test datasets.

        Splits are assembled into per-dataset ``subset`` dictionaries.
        Datetime-mode splits are passed through unchanged, while ratio-mode
        splits are first resolved against the zarr coordinate values and then
        converted to inclusive coordinate ranges before dataset instantiation.

        Parameters
        ----------
        stage : str | None, optional
            Lightning stage hint such as ``"fit"`` or ``"test"``. The value
            is accepted for framework compatibility and is otherwise unused.
        """
        subset_per_split: dict[str, dict[str, Any] | None] = {
            split_name: {} if any(split_name in coord_splits for coord_splits in self.splits.values()) else None
            for split_name in ("train", "val", "test")
        }

        for coord, coord_splits in self.splits.items():
            if splitting_uses_tuple_ranges(coord_splits):
                # tuple-based splits are expected to present the start and end
                # of each split, and so are passed through directly as the
                # subset values for each split
                coord_values_per_split: dict[str, tuple[str, str] | None] = {
                    "train": coord_splits["train"],
                    "val": coord_splits["val"],
                    "test": coord_splits.get("test"),
                }
            elif splitting_uses_fractions(coord_splits):
                # for ratio-based splits, the splitting start-end range tuples
                # are constructed by breaking up the given coordinate in
                # successive segments (the succession is defined from the order
                # of the keys in the splits dict)
                coord_values_per_split = compute_split_ranges_from_splitting_ratios(
                    self.dataset_factory, coord, coord_splits
                )
            else:
                raise NotImplementedError(f"Unsupported split mode for coordinate {coord!r}: {coord_splits!r}")

            for split_name, split_val in coord_values_per_split.items():
                if split_val is None:
                    subset_per_split[split_name] = None
                elif subset_per_split[split_name] is not None:
                    subset_per_split[split_name][coord] = split_val

        augment_flags = {"train": True, "val": False, "test": False}
        for split in ("train", "val", "test"):
            subset = subset_per_split[split]
            if subset is None:
                setattr(self, f"{split}_dataset", None)
            else:
                setattr(
                    self,
                    f"{split}_dataset",
                    self.dataset_factory(subset=subset, augment=augment_flags[split]),
                )

        logger.info("{}.setup() complete, containing:", self.__class__.__name__)
        for split in ("train", "val", "test"):
            dataset = getattr(self, f"{split}_dataset", None)
            if dataset is not None:
                logger.info(
                    "  {:5s}: {:>6d} samples, subset={}",
                    split,
                    len(dataset),
                    subset_per_split[split],
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

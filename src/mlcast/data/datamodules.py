"""PyTorch Lightning data modules for forecasting and reconstruction tasks."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from mlcast.data.forecasting import ForecastingDataset
from mlcast.data.reconstruction import ReconstructionDataset
from mlcast.data.splits import (
    compute_split_ranges_from_splitting_ratios,
    splitting_uses_fractions,
    splitting_uses_tuple_ranges,
    validate_splits,
)


class _BaseDataModule(pl.LightningDataModule):
    """Shared split/build logic for task-level data modules.

    Parameters
    ----------
    sequence_dataset_factory : Callable[..., Dataset]
        Factory that builds source-data sequence datasets and accepts
        ``subset`` and ``augment`` keyword arguments.
    splits : dict of str to dict
        Nested mapping describing train/validation/test coordinate splits.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader``.
    """

    def __init__(
        self,
        sequence_dataset_factory: Callable[..., Dataset],
        splits: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.sequence_dataset_factory = sequence_dataset_factory
        self.splits = splits
        self.dataloader_kwargs = dataloader_kwargs
        validate_splits(self.splits)

    def _build_sequence_dataset(self, subset: dict[str, Any], augment: bool) -> Dataset:
        """Build a source-data sequence dataset for one split.

        Parameters
        ----------
        subset : dict of str to Any
            Coordinate subset passed to the sequence dataset factory.
        augment : bool
            Whether this split should apply data augmentation.

        Returns
        -------
        Dataset
            Built source-data sequence dataset.
        """
        return self.sequence_dataset_factory(subset=subset, augment=augment)

    def _wrap_sequence_dataset(self, base_sequence_dataset: Dataset) -> Dataset:
        """Wrap a sequence dataset into a task-specific dataset.

        Parameters
        ----------
        base_sequence_dataset : Dataset
            Source-data sequence dataset for a split.

        Returns
        -------
        Dataset
            Task-specific dataset for the split.
        """
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        """Create train, validation, and test datasets.

        Parameters
        ----------
        stage : str or None, optional
            Lightning setup stage. Supports ``"fit"``, ``"validate"``,
            ``"test"``, and ``None``. Default is ``None``.

        Raises
        ------
        ValueError
            If ``stage`` is unsupported.
        NotImplementedError
            If a configured split mode is unsupported.
        """
        if stage == "fit":
            requested_splits = {"train", "val"}
        elif stage == "validate":
            requested_splits = {"val"}
        elif stage == "test":
            requested_splits = {"test"}
        elif stage is None:
            requested_splits = {"train", "val", "test"}
        else:
            raise ValueError(f"Unsupported LightningDataModule setup stage: {stage!r}")

        subset_per_split: dict[str, dict[str, Any] | None] = {
            split_name: (
                {}
                if split_name in requested_splits
                and any(split_name in coord_splits for coord_splits in self.splits.values())
                else None
            )
            for split_name in ("train", "val", "test")
        }

        for coord, coord_splits in self.splits.items():
            if splitting_uses_tuple_ranges(coord_splits):
                coord_values_per_split: dict[str, tuple[str, str] | None] = {
                    "train": coord_splits["train"],
                    "val": coord_splits["val"],
                    "test": coord_splits.get("test"),
                }
            elif splitting_uses_fractions(coord_splits):
                coord_values_per_split = compute_split_ranges_from_splitting_ratios(
                    self.sequence_dataset_factory, coord, coord_splits
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
                base_sequence_dataset = self._build_sequence_dataset(subset=subset, augment=augment_flags[split])
                setattr(self, f"{split}_dataset", self._wrap_sequence_dataset(base_sequence_dataset))

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
        """Return the training DataLoader.

        Returns
        -------
        DataLoader
            Training dataloader with shuffled samples.
        """
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader.

        Returns
        -------
        DataLoader
            Validation dataloader without shuffling.
        """
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader.

        Returns
        -------
        DataLoader
            Test dataloader without shuffling.
        """
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)


class ForecastingDataModule(_BaseDataModule):
    """Lightning data module for forecasting datasets.

    Parameters
    ----------
    sequence_dataset_factory : Callable[..., Dataset]
        Factory that builds source-data sequence datasets.
    input_steps : int
        Number of input timesteps in each forecasting sample.
    forecast_steps : int
        Number of target timesteps in each forecasting sample.
    return_mask : bool
        Whether forecasting samples should include ``target_mask``.
    splits : dict of str to dict
        Nested mapping describing train/validation/test coordinate splits.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader``.
    """

    def __init__(
        self,
        sequence_dataset_factory: Callable[..., Dataset],
        input_steps: int,
        forecast_steps: int,
        return_mask: bool,
        splits: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__(sequence_dataset_factory=sequence_dataset_factory, splits=splits, **dataloader_kwargs)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.return_mask = return_mask

    def _wrap_sequence_dataset(self, base_sequence_dataset: Dataset) -> Dataset:
        """Wrap a sequence dataset as a forecasting dataset.

        Parameters
        ----------
        base_sequence_dataset : Dataset
            Sequence dataset for one split.

        Returns
        -------
        Dataset
            Forecasting dataset for the split.
        """
        return ForecastingDataset(
            base_sequence_dataset=base_sequence_dataset,
            input_steps=self.input_steps,
            forecast_steps=self.forecast_steps,
            return_mask=self.return_mask,
        )


class ReconstructionDataModule(_BaseDataModule):
    """Lightning data module for reconstruction datasets.

    Parameters
    ----------
    sequence_dataset_factory : Callable[..., Dataset]
        Factory that builds source-data sequence datasets.
    input_steps : int
        Number of timesteps in each reconstruction window.
    splits : dict of str to dict
        Nested mapping describing train/validation/test coordinate splits.
    **dataloader_kwargs : Any
        Additional keyword arguments forwarded to ``DataLoader``.
    """

    def __init__(
        self,
        sequence_dataset_factory: Callable[..., Dataset],
        input_steps: int,
        splits: dict[str, dict[str, Any]],
        **dataloader_kwargs: Any,
    ) -> None:
        super().__init__(sequence_dataset_factory=sequence_dataset_factory, splits=splits, **dataloader_kwargs)
        self.input_steps = input_steps

    def _wrap_sequence_dataset(self, base_sequence_dataset: Dataset) -> Dataset:
        """Wrap a sequence dataset as a reconstruction dataset.

        Parameters
        ----------
        base_sequence_dataset : Dataset
            Sequence dataset for one split.

        Returns
        -------
        Dataset
            Reconstruction dataset for the split.
        """
        return ReconstructionDataset(base_sequence_dataset=base_sequence_dataset, input_steps=self.input_steps)

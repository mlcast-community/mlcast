"""Contains the dataloader for the MLCast."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from mlcast.data.zarr_dataset import ZarrDataset


@dataclass
class ZarrDataModule(LightningDataModule):
    """`LightningDataModule` for MLCast."""

    dataset_dir: Path | None = None
    variable_name: str = None
    train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1)
    batch_size: int = 8
    dataset_size: int | None = None
    crop_size: int | None = 256
    input_size: int = 5
    output_size: int = 10
    num_workers: int = 0

    pin_memory: bool = False
    data_train: Dataset | None = None
    data_val: Dataset | None = None
    data_test: Dataset | None = None

    def __post_init__(
        self,
    ) -> None:
        """Initialize the LightningDataModule."""
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things
        like random split twice!
        Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once
        the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or
        `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ZarrDataset(
                dataset_dir=self.dataset_dir,
                variable_name=self.variable_name,
                input_size=self.input_size,
                output_size=self.output_size,
                crop_size=self.crop_size,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

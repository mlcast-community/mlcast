from dataclasses import dataclass

import fiddle as fdl
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset as TorchDataset

from ..data.source_data_module import SourceDataDataModule
from ..data.source_datasets import SourceDataPrecomputedSamplingDataset
from ..models.convgru import ConvGruModel
from ..nowcasting_module import NowcastLightningModule


@dataclass
class Experiment:
    """Container for Lightning module, data module, and trainer."""

    pl_module: pl.LightningModule
    data: pl.LightningDataModule
    trainer: pl.Trainer

    def run(self) -> None:
        """Train and evaluate the configured model."""
        self.trainer.fit(self.pl_module, datamodule=self.data)
        self.trainer.test(self.pl_module, datamodule=self.data)


@fiddle.experimental.auto_config.auto_config
def training_experiment() -> Experiment:
    dataset_factory: fdl.Partial[TorchDataset] = fdl.Partial(
        SourceDataPrecomputedSamplingDataset,
        zarr_path="./data/radar.zarr",
        csv_path="./data/sampled_datacubes.csv",
        standard_names=["rainfall_rate"],
        steps=18,
        return_mask=True,
        deterministic=False,
    )

    data: pl.LightningDataModule = SourceDataDataModule(
        ...,
        dataset_factory=dataset_factory,
        train_ratio=0.70,
        batch_size=16,
    )

    network: torch.nn.Module = ConvGruModel(
        ...,
        input_channels=1,
        num_blocks=5,
    )

    pl_module: pl.LightningModule = NowcastLightningModule(
        ...,
        network=network,
        forecast_steps=12,
        ensemble_size=2,
        optimizer=fdl.Partial(torch.optim.Adam),
        lr_scheduler=fdl.Partial(torch.optim.lr_scheduler.ReduceLROnPlateau),
    )

    trainer = pl.Trainer(
        ...,
        accelerator="auto",
        max_epochs=100,
    )

    return Experiment(
        pl_module=pl_module,
        data=data,
        trainer=trainer,
    )

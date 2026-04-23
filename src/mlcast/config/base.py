"""Base Fiddle experiment definitions."""

from dataclasses import dataclass

import fiddle as fdl
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.source_data_module import SourceDataDataModule
from ..data.source_datasets import SourceDataPrecomputedSamplingDataset
from ..models.convgru_modules import ConvGruModel
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
    """Build a Fiddle config for ConvGRU ensemble radar nowcasting.

    This is decorated as a Fiddle ``@auto_config`` function: calling it
    returns a buildable config graph where any parameter can be overridden
    before instantiation via ``fdl.build()``.

    Returns
    -------
    Experiment
        Configured experiment with model, data, and trainer.
    """
    dataset_factory = fdl.Partial(
        SourceDataPrecomputedSamplingDataset,
        zarr_path="./data/radar.zarr",
        csv_path="./data/sampled_datacubes.csv",
        standard_names=["rainfall_rate"],
        steps=18,
        return_mask=True,
        deterministic=False,
    )

    data = SourceDataDataModule(
        dataset_factory=dataset_factory,
        train_ratio=0.70,
        val_ratio=0.15,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )

    network = ConvGruModel(
        input_channels=1,
        num_blocks=5,
        noisy_decoder=False,
    )

    pl_module = NowcastLightningModule(
        network=network,
        forecast_steps=12,
        ensemble_size=2,
        loss_class="crps",
        loss_params={"temporal_lambda": 0.01},
        masked_loss=True,
        optimizer=fdl.Partial(torch.optim.Adam, lr=1e-4, fused=True),
        lr_scheduler=fdl.Partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.5, patience=10),
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            ModelCheckpoint(monitor="train_loss_epoch", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=100, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="mlcast"),
    )

    return Experiment(
        pl_module=pl_module,
        data=data,
        trainer=trainer,
    )

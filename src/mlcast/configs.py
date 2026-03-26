"""Fiddle-based experiment configuration factories.

Uses ``@auto_config`` to define experiment factories that return buildable
Fiddle config graphs. Override any parameter before calling ``fdl.build()``.
"""

from dataclasses import dataclass

import fiddle as fdl
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .data.zarr_datamodule import RadarDataModule
from .models.convgru import RadarLightningModel

__all__ = [
    "Experiment",
    "convgru_experiment",
]


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
def convgru_experiment(
    *,
    zarr_path: str = "./data/radar.zarr",
    csv_path: str = "./data/sampled_datacubes.csv",
    variable_name: str = "RR",
) -> Experiment:
    """Build a Fiddle config for ConvGRU ensemble radar nowcasting.

    This is decorated as a Fiddle ``@auto_config`` function: calling it
    returns a buildable config graph where any parameter can be overridden
    before instantiation via ``fdl.build()``.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the sampled datacubes CSV.
    variable_name : str
        Name of the rain rate variable in the Zarr store.

    Returns
    -------
    Experiment
        Configured experiment with model, data, and trainer.
    """
    data = RadarDataModule(
        zarr_path=zarr_path,
        csv_path=csv_path,
        variable_name=variable_name,
        steps=18,
        train_ratio=0.70,
        val_ratio=0.15,
        return_mask=True,
        deterministic=False,
        augment=True,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )

    pl_module = RadarLightningModel(
        input_channels=1,
        forecast_steps=12,
        num_blocks=5,
        ensemble_size=2,
        noisy_decoder=False,
        loss_class="crps",
        loss_params={"temporal_lambda": 0.01},
        masked_loss=True,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-4, "fused": True},
        lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_params={"mode": "min", "factor": 0.5, "patience": 10},
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
        logger=TensorBoardLogger(save_dir="logs", name="convgru"),
    )

    return Experiment(
        pl_module=pl_module,
        data=data,
        trainer=trainer,
    )


def config_to_dict(cfg: fdl.Config) -> dict:
    """Recursively convert a Fiddle config to a nested dictionary."""
    result = {}
    for key, value in fdl.ordered_arguments(cfg).items():
        result[key] = config_to_dict(value) if isinstance(value, fdl.Config) else value
    return result


def train_from_config(cfg: fdl.Config) -> None:
    """Build and run an experiment from a Fiddle config.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration as returned by :func:`convgru_experiment`.
    """
    experiment = fdl.build(cfg)
    experiment.run()

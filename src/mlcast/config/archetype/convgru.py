"""ConvGRU ensemble nowcasting experiment configuration."""

import fiddle as fdl
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlcast.data.datamodules import ForecastingDataModule
from mlcast.data.sequence import SourceDataPrecomputedSequenceDataset
from mlcast.models.convgru import ConvGruModel
from mlcast.modules.forecasting import OutputSpaceForecastingTaskModule

from ..base import Experiment


@fiddle.experimental.auto_config.auto_config
def convgru_training_experiment() -> Experiment:
    """Build a Fiddle config for ConvGRU ensemble radar nowcasting.

    This is decorated as a Fiddle ``@auto_config`` function: calling it
    returns a buildable config graph where any parameter can be overridden
    before instantiation via ``fdl.build()``.

    Returns
    -------
    Experiment
        Configured experiment with model, data, and trainer.
    """
    sequence_dataset_factory = fdl.Partial(
        SourceDataPrecomputedSequenceDataset,
        zarr_path="./data/radar.zarr",
        csv_path="./data/sampled_datacubes.csv",
        standard_names=["rainfall_rate"],
        sequence_steps=18,
        deterministic=False,
    )

    data = ForecastingDataModule(
        sequence_dataset_factory=sequence_dataset_factory,
        input_steps=6,
        forecast_steps=12,
        return_mask=True,
        splits={"time": {"train": 0.70, "val": 0.15, "test": 0.15}},
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )

    network = ConvGruModel(
        input_steps=6,
        forecast_steps=12,
        ensemble_size=2,
        input_channels=1,
        num_blocks=5,
        noisy_decoder=False,
    )

    pl_module = OutputSpaceForecastingTaskModule(
        network=network,
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
            ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min"),
            ModelCheckpoint(monitor="train/loss_epoch", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val/loss", patience=100, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="mlcast"),
    )

    return Experiment(
        pl_module=pl_module,
        data=data,
        trainer=trainer,
    )

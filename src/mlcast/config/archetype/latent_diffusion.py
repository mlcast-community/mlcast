"""Fiddle configuration for two-stage latent diffusion training."""

from dataclasses import dataclass

import fiddle as fdl
import fiddle.experimental.auto_config
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mlcast.data.datamodules import ForecastingDataModule, ReconstructionDataModule
from mlcast.data.sequence import SourceDataPrecomputedSequenceDataset
from mlcast.models.autoencoder import AutoencoderNet, Decoder, Encoder
from mlcast.models.diffusion import ConditionerNet, DenoiserUNet, DiffusionScheduler, LatentDiffusionNet
from mlcast.modules.forecasting import LatentDiffusionTaskModule
from mlcast.modules.reconstruction import ReconstructionTaskModule

from ..base import Experiment


@dataclass
class LatentDiffusionTrainingExperiment:
    """Two-stage latent diffusion training experiment.

    Parameters
    ----------
    stage1 : Experiment
        Reconstruction training stage for the autoencoder.
    stage2 : Experiment
        Latent diffusion training stage reusing the same trained autoencoder
        instance from stage 1.
    """

    stage1: Experiment
    stage2: Experiment

    @property
    def trainer(self) -> pl.Trainer:
        """Expose the first trainer for orchestrator compatibility.

        Returns
        -------
        pl.Trainer
            Trainer used by stage 1.
        """
        return self.stage1.trainer

    def run(self) -> None:
        """Run stage-1 reconstruction followed by stage-2 latent diffusion."""
        self.stage1.trainer.fit(self.stage1.pl_module, datamodule=self.stage1.data)
        self.stage1.trainer.test(self.stage1.pl_module, datamodule=self.stage1.data)

        self.stage2.trainer.fit(self.stage2.pl_module, datamodule=self.stage2.data)
        self.stage2.trainer.test(self.stage2.pl_module, datamodule=self.stage2.data)


@fiddle.experimental.auto_config.auto_config
def latent_diffusion_experiment() -> LatentDiffusionTrainingExperiment:
    """Build a Fiddle config for two-stage latent diffusion training.

    Returns
    -------
    LatentDiffusionTrainingExperiment
        Configured two-stage experiment with shared autoencoder identity across
        reconstruction and latent diffusion stages.
    """
    input_steps = 4
    forecast_steps = 12
    sequence_steps = input_steps + forecast_steps

    sequence_dataset_factory = fdl.Partial(
        SourceDataPrecomputedSequenceDataset,
        zarr_path="./data/radar.zarr",
        csv_path="./data/sampled_datacubes.csv",
        standard_names=["rainfall_rate"],
        sequence_steps=sequence_steps,
        deterministic=False,
    )

    autoencoder = AutoencoderNet(
        encoder=Encoder(input_channels=1, hidden_channels=16, latent_channels=32, num_blocks=2),
        decoder=Decoder(output_channels=1, hidden_channels=16, latent_channels=32, num_blocks=2),
    )

    stage1_data = ReconstructionDataModule(
        sequence_dataset_factory=sequence_dataset_factory,
        input_steps=input_steps,
        splits={"time": {"train": 0.70, "val": 0.15, "test": 0.15}},
        batch_size=4,
        num_workers=8,
        pin_memory=True,
    )
    stage1_module = ReconstructionTaskModule(
        network=autoencoder,
        loss_class="mse",
        optimizer=fdl.Partial(torch.optim.AdamW, lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3, fused=True),
        lr_scheduler=fdl.Partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.25, patience=3),
    )
    stage1_trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=20,
        accumulate_grad_batches=2,
        callbacks=[
            ModelCheckpoint(monitor="val/rec_loss", save_top_k=3, mode="min"),
            EarlyStopping(monitor="val/rec_loss", patience=6, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="mlcast_latent_diffusion_stage1"),
    )

    stage2_data = ForecastingDataModule(
        sequence_dataset_factory=sequence_dataset_factory,
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        return_mask=False,
        splits={"time": {"train": 0.70, "val": 0.15, "test": 0.15}},
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )
    diffusion_net = LatentDiffusionNet(
        conditioner=ConditionerNet(latent_channels=32, hidden_channels=32, num_blocks=2),
        denoiser=DenoiserUNet(latent_channels=32, condition_channels=32, hidden_channels=32, num_blocks=2),
        scheduler=DiffusionScheduler(timesteps=1000),
    )
    stage2_module = LatentDiffusionTaskModule(
        autoencoder=autoencoder,
        diffusion_net=diffusion_net,
        forecast_steps=forecast_steps,
        ensemble_size=2,
        optimizer=fdl.Partial(torch.optim.AdamW, lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-3, fused=True),
        lr_scheduler=fdl.Partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min", factor=0.25, patience=3),
        ema_decay=0.9999,
    )
    stage2_trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=20,
        accumulate_grad_batches=2,
        callbacks=[
            ModelCheckpoint(monitor="val/loss", save_top_k=3, mode="min"),
            EarlyStopping(monitor="val/loss", patience=6, mode="min", check_finite=False),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="mlcast_latent_diffusion_stage2"),
    )

    return LatentDiffusionTrainingExperiment(
        stage1=Experiment(pl_module=stage1_module, data=stage1_data, trainer=stage1_trainer),
        stage2=Experiment(pl_module=stage2_module, data=stage2_data, trainer=stage2_trainer),
    )

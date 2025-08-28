"""Abstract base classes for nowcasting models.

This module provides the core API for machine learning-based precipitation
nowcasting models that all models must implement.
"""

import abc
from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import xarray as xr
from torch import nn


class NowcastingModelBase(abc.ABC):
    """Abstract base class for precipitation nowcasting models.

    This class defines the standard interface that all nowcasting models
    must implement. It provides a consistent API for training, prediction,
    and model persistence across different ML approaches.

    Attributes:
        timestep_length: Time resolution of the model's predictions
    """

    timestep_length: np.timedelta64 | None = None
    PLModuleClass: L.LightningModule | None = None

    def __init__(self):
        """Initialize the nowcasting model."""
        self.pl_module = self.PLModuleClass() if self.PLModuleClass is not None else None

    @abc.abstractmethod
    def save(self, path: str, **kwargs: Any) -> None:
        """Save the trained model to disk.

        Args:
            path: File path where the model should be saved
            **kwargs: Additional arguments for model saving

        Note:
            This method needs to be implemented in concrete subclasses
            to handle model serialization.
        """
        pass

    @abc.abstractmethod
    def load(self, path: str, **kwargs: Any) -> None:
        """Load a pre-trained model from disk.

        Args:
            path: File path to the saved model
            **kwargs: Additional arguments for model loading

        Note:
            This method needs to be implemented in concrete subclasses
            to handle model deserialization.
        """
        pass

    @abc.abstractmethod
    def fit(self, da_rr: xr.DataArray, **kwargs: Any) -> None:
        """Train the nowcasting model on precipitation data.

        Args:
            da_rr: xarray DataArray containing precipitation radar data
                with time, latitude, and longitude dimensions
            **kwargs: Additional arguments for training (e.g., batch_size, epochs)

        Note:
            Concrete implementations should:
            1. Process the input data (scaling, temporal windowing)
            2. Train the underlying ML model
            3. Store scaling parameters and timestep information
        """
        pass

    @abc.abstractmethod
    def predict(self, da_rr: xr.DataArray, duration: str, **kwargs: Any) -> xr.DataArray:
        """Generate precipitation forecasts.

        Args:
            da_rr: xarray DataArray containing initial precipitation conditions
            duration: ISO 8601 duration string (e.g., "PT1H" for 1 hour)
                specifying how far into the future to predict
            **kwargs: Additional arguments for prediction (e.g., batch_size, device)

        Returns:
            xarray DataArray containing precipitation predictions with
            original spatial dimensions plus an "elapsed_time" dimension

        Note:
            Concrete implementations should:
            1. Process input data using stored scaling parameters
            2. Generate predictions using the trained model
            3. Return results in the original coordinate system
        """
        pass


class NowcastingLightningModule(L.LightningModule):
    """Base class for PyTorch Lightning modules used in nowcasting models.

    This class provides a standard interface for training and validation
    steps, as well as optimizer configuration.
    """

    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        optimizer_class: Any | None = None,
        optimizer_kwargs: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss"])
        self.net = net
        self.loss = loss
        self.optimizer_class = torch.optim.Adam if optimizer_class is None else optimizer_class

    def forward(self, x: torch.Tensor, n_timesteps: int) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x: Input tensor with shape (batch, seq_len, channels, height, width)
            n_timesteps: Number of timesteps to predict
        Returns:
            Output tensor with shape (batch, n_timesteps, channels, height, width)
        """
        return self.net(x, n_timesteps)  # Assuming net is a callable model

    def model_step(self, batch: Any, batch_idx: int, step_name: str = "train") -> torch.Tensor:
        """Generic model step for training or validation.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        x, y = batch
        predictions = self.forward(x, n_timesteps=y.shape[1])
        loss = self.loss(predictions, y)
        if isinstance(loss, dict):
            # append step name to loss keys for logging
            loss = {f"{step_name}/{k}": v.item() for k, v in loss}
            self.log_dict(loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss.get("loss", loss.get("total_loss", None))
            if loss is None:
                raise ValueError(f"Loss is None for step {step_name}. Ensure loss function returns a valid tensor.")
        else:
            self.log(f"{step_name}/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            Optimizer instance to use for training
        """
        return self.optimizer_class(self.parameters(), **(self.hparams.optimizer_kwargs or {}))

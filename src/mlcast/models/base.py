"""Generic Lightning module for radar precipitation nowcasting.

Wraps an injected PyTorch :class:`nn.Module` (the network architecture) and
handles training, validation, and test steps including loss computation,
ensemble generation, and TensorBoard image logging.
"""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.data.normalization import DENORMALIZATION_REGISTRY, NORMALIZATION_REGISTRY
from mlcast.losses import build_loss
from mlcast.visualization import log_images


class NowcastLightningModule(pl.LightningModule):
    """Generic PyTorch Lightning module for nowcasting.

    Wraps an injected PyTorch `nn.Module` (the network architecture) and
    handles training, validation, test steps, loss computation, ensemble
    generation, and TensorBoard logging.

    Parameters
    ----------
    network : torch.nn.Module
        The PyTorch network architecture to train.
    ensemble_size : int, optional
        Number of ensemble members to generate. Default is ``1``.
    forecast_steps : int or None, optional
        Number of future timesteps to forecast. Default is ``None``.
    loss_class : type[torch.nn.Module] or str, optional
        Loss function class or its string name. Default is ``"mse"``.
    loss_params : dict or None, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    masked_loss : bool, optional
        Whether to wrap the loss with :class:`MaskedLoss`. Default is ``False``.
    optimizer_class : type or None, optional
        Optimizer class. Default is ``None`` (Adam).
    optimizer_params : dict or None, optional
        Keyword arguments for the optimizer. Default is ``None``.
    lr_scheduler_class : type or None, optional
        Learning rate scheduler class. Default is ``None``.
    lr_scheduler_params : dict or None, optional
        Keyword arguments for the LR scheduler. Default is ``None``.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        ensemble_size: int = 1,
        forecast_steps: type | int | None = None,
        loss_class: type[torch.nn.Module] | str = "mse",
        loss_params: dict[str, Any] | None = None,
        masked_loss: bool = False,
        optimizer_class: type[torch.optim.Optimizer] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        lr_scheduler_class: type | None = None,
        lr_scheduler_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["network"])

        self.network = network

        self.criterion = build_loss(
            loss_class=self.hparams["loss_class"],
            loss_params=self.hparams["loss_params"],
            masked_loss=self.hparams["masked_loss"],
        )
        self.log_images_iterations = [50, 100, 200, 500, 750, 1000, 2000, 5000]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
        forecast_steps: int,
        ensemble_size: int | None = None,
    ) -> Float[torch.Tensor, "batch forecast_steps out_channels height width"]:
        """Run the network forward pass.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input tensor.
        forecast_steps : int
            Number of steps to forecast.
        ensemble_size : int or None, optional
            Number of ensemble members to generate. If ``None``, uses the initialized value. Default is ``None``.

        Returns
        -------
        preds : Float[torch.Tensor, "batch forecast_steps out_channels height width"]
            Forecast tensor.
        """
        ensemble_size = self.hparams["ensemble_size"] if ensemble_size is None else ensemble_size
        return self.network(x, steps=forecast_steps, ensemble_size=ensemble_size)

    def shared_step(
        self, batch: dict[str, torch.Tensor], split: str = "train", ensemble_size: int | None = None
    ) -> torch.Tensor:
        """Shared forward step for training, validation, and testing.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data. Must contain the
            key ``"data"`` and optionally ``"mask"`` if ``masked_loss`` is ``True``.
        split : str, optional
            The data split being processed (e.g., ``"train"``, ``"val"``, ``"test"``).
            Used for logging. Default is ``"train"``.
        ensemble_size : int or None, optional
            The number of ensemble members to generate. If ``None``, uses the
            default from hyper-parameters. Default is ``None``.

        Returns
        -------
        loss : torch.Tensor
            The computed loss for the batch.
        """
        data = batch["data"]
        past = data[:, : -self.hparams["forecast_steps"]]
        future = data[:, -self.hparams["forecast_steps"] :]

        preds = self(past, forecast_steps=self.hparams["forecast_steps"], ensemble_size=ensemble_size).clamp(
            min=-1, max=1
        )

        if self.hparams["masked_loss"]:
            mask = batch["mask"][:, -self.hparams["forecast_steps"] :]
            loss = self.criterion(preds, future, mask)
        else:
            loss = self.criterion(preds, future)

        if isinstance(loss, tuple):
            loss, log_dict = loss
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)

        if self.hparams["ensemble_size"] > 1:
            ensemble_std = preds.std(dim=2).mean()
            self.log(f"{split}_ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

        if (
            split == "train"
            and self.logger is not None
            and getattr(self.logger, "experiment", None) is not None
            and (
                self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0
            )
        ):
            log_images(
                past=past,
                future=future,
                preds=preds,
                logger_experiment=self.logger.experiment,  # type: ignore
                global_step=self.global_step,
                ensemble_size=self.hparams["ensemble_size"],
                split=split,
            )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The training loss.
        """
        return self.shared_step(batch, split="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The validation loss.
        """
        return self.shared_step(batch, split="val", ensemble_size=10)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute a single test step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            A dictionary containing the batched input data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        loss : torch.Tensor
            The test loss.
        """
        return self.shared_step(batch, split="test", ensemble_size=10)

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and optional learning rate scheduler.

        Returns
        -------
        config : dict of str to Any
            A dictionary containing the instantiated ``"optimizer"`` and
            optionally ``"lr_scheduler"`` configurations for PyTorch Lightning.
        """
        if self.hparams["optimizer_class"] is not None:
            optimizer_class = self.hparams["optimizer_class"]
            optimizer = (
                optimizer_class(self.parameters(), **self.hparams["optimizer_params"])
                if self.hparams["optimizer_params"] is not None
                else optimizer_class(self.parameters())
            )
        else:
            optimizer = torch.optim.Adam(self.parameters())

        if self.hparams["lr_scheduler_class"] is not None:
            lr_scheduler_class = self.hparams["lr_scheduler_class"]
            lr_scheduler = (
                lr_scheduler_class(optimizer, **self.hparams["lr_scheduler_params"])
                if self.hparams["lr_scheduler_params"] is not None
                else lr_scheduler_class(optimizer)
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
        else:
            return {"optimizer": optimizer}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "NowcastLightningModule":
        """Load a model from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to the saved PyTorch Lightning checkpoint (``.ckpt``) file.
        device : str, optional
            The device to map the model weights to (e.g., ``"cpu"`` or ``"cuda"``).
            Default is ``"cpu"``.

        Returns
        -------
        model : NowcastLightningModule
            The loaded PyTorch Lightning model instance.
        """
        return cls.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(device),
            strict=True,
            weights_only=False,
        )

    def predict(
        self,
        past: torch.Tensor,
        forecast_steps: int = 1,
        ensemble_size: int | None = 1,
        standard_name: str = "rainfall_rate",
    ) -> np.ndarray[Any, Any]:
        """Generate precipitation forecasts from past radar observations.

        Input should be raw unnormalized values.

        Parameters
        ----------
        past : torch.Tensor
            Past radar frames as unnormalized values (e.g., mm/h or kg m-2 s-1), of shape ``(T, H, W)``.
        forecast_steps : int, optional
            Number of future timesteps to forecast. Default is ``1``.
        ensemble_size : int, optional
            Number of ensemble members. Default is ``1``.
        standard_name : str, optional
            The CF standard name defining the input/output domain for normalization lookup.
            Default is ``"rainfall_rate"``.

        Returns
        -------
        preds : np.ndarray
            Forecasted unnormalized values, of shape
            ``(ensemble_size, forecast_steps, H, W)``.
        """
        if len(past.shape) != 3:
            raise ValueError("Input must be of shape (T, H, W)")

        T, H, W = past.shape
        ensemble_size = self.hparams["ensemble_size"] if ensemble_size is None else ensemble_size

        past_clean = np.nan_to_num(past.cpu().numpy())
        past_clean = past_clean[np.newaxis, :, np.newaxis, ...]

        norm_func = NORMALIZATION_REGISTRY[standard_name]
        norm_past = norm_func(past_clean)

        x = torch.from_numpy(norm_past)
        x = x.to(self.device)

        self.eval()
        with torch.no_grad():
            preds_tensor = self.network(x, steps=forecast_steps, ensemble_size=ensemble_size)

        preds_np: np.ndarray[Any, Any] = preds_tensor.cpu().numpy()

        denorm_func = DENORMALIZATION_REGISTRY[standard_name]
        preds_np = denorm_func(preds_np)

        preds_np = preds_np.squeeze(0)
        preds_np = np.swapaxes(preds_np, 0, 1)

        return preds_np

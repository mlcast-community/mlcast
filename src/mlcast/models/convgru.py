"""Concrete Lightning module for ConvGRU-based radar precipitation nowcasting.

Wraps the :class:`EncoderDecoder` model and handles training, validation,
and test steps including loss computation, ensemble generation, and
TensorBoard image logging.
"""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from mlcast.losses import build_loss
from mlcast.modules.convgru_modules import EncoderDecoder
from mlcast.utils import normalized_to_rainrate, rainrate_to_normalized


def apply_radar_colormap(tensor: torch.Tensor) -> torch.Tensor:
    """Convert grayscale radar values to RGB using the STEPS-BE colorscale.

    Maps normalized values in [0, 1] (representing 0-60 dBZ) to a 14-color
    discrete colormap. Pixels below 10 dBZ are rendered as white.

    Parameters
    ----------
    tensor : torch.Tensor
        Grayscale tensor with values in [0, 1], of shape ``(N, 1, H, W)``.

    Returns
    -------
    rgb : torch.Tensor
        RGB tensor of shape ``(N, 3, H, W)`` with values in [0, 1].
    """
    colors = (
        torch.tensor(
            [
                [0, 255, 255],
                [0, 191, 255],
                [30, 144, 255],
                [0, 0, 255],
                [127, 255, 0],
                [50, 205, 50],
                [0, 128, 0],
                [0, 100, 0],
                [255, 255, 0],
                [255, 215, 0],
                [255, 165, 0],
                [255, 0, 0],
                [255, 0, 255],
                [139, 0, 139],
            ],
            dtype=torch.float32,
            device=tensor.device,
        )
        / 255.0
    )

    num_colors = len(colors)
    min_dbz_norm = 10 / 60
    max_dbz_norm = 1.0
    thresholds = torch.linspace(min_dbz_norm, max_dbz_norm, num_colors + 1, device=tensor.device)

    N, _, H, W = tensor.shape
    output = torch.ones(N, 3, H, W, dtype=torch.float32, device=tensor.device)

    for i in range(num_colors - 1):
        mask = (tensor[:, 0] >= thresholds[i]) & (tensor[:, 0] < thresholds[i + 1])
        for c in range(3):
            output[:, c][mask] = colors[i, c]

    mask = tensor[:, 0] >= thresholds[num_colors - 1]
    for c in range(3):
        output[:, c][mask] = colors[-1, c]

    return output


class RadarLightningModel(pl.LightningModule):
    """PyTorch Lightning module for radar precipitation nowcasting.

    Wraps an :class:`EncoderDecoder` model and handles training, validation,
    and test steps including loss computation, ensemble generation, and
    TensorBoard image logging.

    Parameters
    ----------
    input_channels : int
        Number of input channels per grid point.
    num_blocks : int
        Number of encoder/decoder blocks in the model.
    ensemble_size : int, optional
        Number of ensemble members to generate. Default is ``1``.
    noisy_decoder : bool, optional
        Whether to use random noise as decoder input. Default is ``False``.
    forecast_steps : int or None, optional
        Number of future timesteps to forecast. Default is ``None``.
    loss_class : type, str, or None, optional
        Loss function class or its string name. Default is ``None`` (MSELoss).
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
        input_channels: int,
        num_blocks: int,
        ensemble_size: int = 1,
        noisy_decoder: bool = False,
        forecast_steps: type | int | None = None,
        loss_class: type | str | None = None,
        loss_params: dict[str, Any] | None = None,
        masked_loss: bool = False,
        optimizer_class: type | None = None,
        optimizer_params: dict[str, Any] | None = None,
        lr_scheduler_class: type | None = None,
        lr_scheduler_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = EncoderDecoder(self.hparams.input_channels, self.hparams.num_blocks)

        self.criterion = build_loss(
            loss_class=self.hparams.loss_class,
            loss_params=self.hparams.loss_params,
            masked_loss=self.hparams.masked_loss,
        )
        self.log_images_iterations = [50, 100, 200, 500, 750, 1000, 2000, 5000]

    def forward(self, x: torch.Tensor, forecast_steps: int, ensemble_size: int | None = None) -> torch.Tensor:
        """Run the encoder-decoder forward pass."""
        ensemble_size = self.hparams.ensemble_size if ensemble_size is None else ensemble_size
        return self.model(
            x, steps=forecast_steps, noisy_decoder=self.hparams.noisy_decoder, ensemble_size=ensemble_size
        )

    def shared_step(
        self, batch: dict[str, torch.Tensor], split: str = "train", ensemble_size: int | None = None
    ) -> torch.Tensor:
        """Shared forward step for training, validation, and testing."""
        data = batch["data"]
        past = data[:, : -self.hparams.forecast_steps]
        future = data[:, -self.hparams.forecast_steps :]

        preds = self(past, forecast_steps=self.hparams.forecast_steps, ensemble_size=ensemble_size).clamp(min=-1, max=1)

        if self.hparams.masked_loss:
            mask = batch["mask"][:, -self.hparams.forecast_steps :]
            loss = self.criterion(preds, future, mask)
        else:
            loss = self.criterion(preds, future)

        if isinstance(loss, tuple):
            loss, log_dict = loss
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)

        if self.hparams.ensemble_size > 1:
            ensemble_std = preds.std(dim=2).mean()
            self.log(f"{split}_ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

        if split == "train" and (
            self.global_step in self.log_images_iterations or self.global_step % self.log_images_iterations[-1] == 0
        ):
            self.log_images(past, future, preds, split=split)
        return loss

    def log_images(self, past: torch.Tensor, future: torch.Tensor, preds: torch.Tensor, split: str = "val") -> None:
        """Log radar image grids to TensorBoard."""
        sample_idx = 0

        past_sample = past[sample_idx]
        if self.hparams.ensemble_size > 1:
            past_sample = past_sample.mean(dim=1, keepdim=True)
        past_norm = (past_sample + 1) / 2
        past_rgb = apply_radar_colormap(past_norm)
        past_grid = torchvision.utils.make_grid(past_rgb, nrow=past_sample.shape[0])
        self.logger.experiment.add_image(f"{split}/past", past_grid, self.global_step)

        future_sample = future[sample_idx]
        preds_sample = preds[sample_idx]

        if self.hparams.ensemble_size > 1:
            preds_avg = preds_sample.mean(dim=1, keepdim=True)
            num_members_to_log = min(3, preds_sample.shape[1])

            rows = [future_sample]
            rows.append(preds_avg)
            for i in range(num_members_to_log):
                rows.append(preds_sample[:, i : i + 1, :, :])

            all_frames = torch.cat(rows, dim=0)
            all_frames_norm = (all_frames + 1) / 2
            all_frames_rgb = apply_radar_colormap(all_frames_norm)
            grid = torchvision.utils.make_grid(all_frames_rgb, nrow=future_sample.shape[0])
            self.logger.experiment.add_image(f"{split}/preds", grid, self.global_step)
        else:
            rows = [future_sample, preds_sample]
            all_frames = torch.cat(rows, dim=0)
            all_frames_norm = (all_frames + 1) / 2
            all_frames_rgb = apply_radar_colormap(all_frames_norm)
            grid = torchvision.utils.make_grid(all_frames_rgb, nrow=future_sample.shape[0])
            self.logger.experiment.add_image(f"{split}/preds", grid, self.global_step)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="val", ensemble_size=10)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, split="test", ensemble_size=10)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and optional learning rate scheduler."""
        if self.hparams.optimizer_class is not None:
            optimizer = (
                self.hparams.optimizer_class(self.parameters(), **self.hparams.optimizer_params)
                if self.hparams.optimizer_params is not None
                else self.hparams.optimizer_class(self.parameters())
            )
        else:
            optimizer = torch.optim.Adam(self.parameters())

        if self.hparams.lr_scheduler_class is not None:
            lr_scheduler = (
                self.hparams.lr_scheduler_class(optimizer, **self.hparams.lr_scheduler_params)
                if self.hparams.lr_scheduler_params is not None
                else self.hparams.lr_scheduler_class(optimizer)
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}
        else:
            return {"optimizer": optimizer}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "RadarLightningModel":
        """Load a model from a checkpoint file."""
        return cls.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(device),
            strict=True,
            weights_only=False,
        )

    def predict(self, past: torch.Tensor, forecast_steps: int = 1, ensemble_size: int | None = 1) -> np.ndarray:
        """Generate precipitation forecasts from past radar observations.

        Input should be raw rain rate values.

        Parameters
        ----------
        past : torch.Tensor
            Past radar frames as rain rate in mm/h, of shape ``(T, H, W)``.
        forecast_steps : int, optional
            Number of future timesteps to forecast. Default is ``1``.
        ensemble_size : int, optional
            Number of ensemble members. Default is ``1``.

        Returns
        -------
        preds : np.ndarray
            Forecasted rain rate in mm/h, of shape
            ``(ensemble_size, forecast_steps, H, W)``.
        """
        if len(past.shape) != 3:
            raise ValueError("Input must be of shape (T, H, W)")

        T, H, W = past.shape
        ensemble_size = self.hparams.ensemble_size if ensemble_size is None else ensemble_size

        divisor = 2 ** (self.hparams.num_blocks)
        padH = (divisor - (H % divisor)) % divisor
        padW = (divisor - (W % divisor)) % divisor
        padded_past = past
        if padH != 0 or padW != 0:
            padded_past = np.pad(past, ((0, 0), (0, padH), (0, padW)), mode="constant", constant_values=0)

        past_clean = np.nan_to_num(padded_past)
        past_clean = past_clean[np.newaxis, :, np.newaxis, ...]
        norm_past = rainrate_to_normalized(past_clean)
        x = torch.from_numpy(norm_past)
        x = x.to(self.device)

        self.eval()
        with torch.no_grad():
            preds = self.model(x, forecast_steps, self.hparams.noisy_decoder, ensemble_size)

        preds = preds.cpu().numpy()
        preds = normalized_to_rainrate(preds)
        preds = preds.squeeze(0)
        preds = np.swapaxes(preds, 0, 1)
        preds = preds[..., :H, :W]

        return preds

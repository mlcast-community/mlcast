"""Forecasting task-level Lightning module wrappers."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped

from mlcast.data.normalization import DENORMALIZATION_REGISTRY, NORMALIZATION_REGISTRY
from mlcast.losses import build_loss
from mlcast.models.autoencoder import AutoencoderNet
from mlcast.models.diffusion.ema import ExponentialMovingAverage
from mlcast.models.diffusion.loss import DiffusionLoss
from mlcast.models.diffusion.net import LatentDiffusionNet
from mlcast.models.diffusion.sampler import DiffusionSampler
from mlcast.visualization import log_images


class BaseForecastingTaskModule(pl.LightningModule):
    """Base Lightning module for forecasting-shaped tasks.

    Purpose
    -------
    This class provides the common PyTorch Lightning plumbing shared by
    forecasting-oriented task modules. It centralizes the optimizer and
    scheduler configuration interface, the train/validation/test step routing,
    and the normalization-aware prediction helper used by forecasting tasks.

    Ownership
    ---------
    This base class owns:

    - optimizer and scheduler factories
    - generic Lightning step orchestration
    - normalization and denormalization logic for ``predict``

    It does not own:

    - a specific forecasting architecture
    - a concrete task loss
    - the choice of which parameters are trainable
    - any task-specific inference logic beyond normalized I/O handling

    Training behavior
    -----------------
    Training, validation, and test steps all delegate to the subclass hook
    :meth:`compute_loss`. Subclasses are also responsible for exposing the
    exact parameter set to optimize through the :attr:`trainable_parameters`
    property.

    Inference behavior
    ------------------
    ``predict`` accepts unnormalized input observations, applies the configured
    normalization for the requested standard name, delegates normalized
    forecasting to :meth:`predict_normalized`, then denormalizes the model
    outputs back to physical units.

    Parameters
    ----------
    optimizer : Callable[..., torch.optim.Optimizer] or None, optional
        Optimizer factory. Default is ``None`` (Adam).
    lr_scheduler : Callable[..., torch.optim.lr_scheduler.LRScheduler] or None, optional
        Learning-rate scheduler factory. Default is ``None``.
    """

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        self.optimizer_factory = optimizer
        self.lr_scheduler_factory = lr_scheduler

    @property
    @abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return the parameters optimized for this forecasting task.

        Returns
        -------
        list of torch.nn.Parameter
            Trainable parameters owned by the concrete forecasting task.
        """

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor], split: str = "train") -> torch.Tensor:
        """Compute and log loss for one forecasting batch.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch.
        split : str, optional
            Current data split. Default is ``"train"``.

        Returns
        -------
        torch.Tensor
            Scalar task loss.
        """

    @jaxtyped(typechecker=beartype)
    def predict_normalized(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
    ) -> Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]:
        """Predict normalized forecasts from normalized inputs.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Normalized forecasting input.

        Returns
        -------
        Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]
            Normalized forecast tensor with an explicit ensemble dimension.
        """
        return self(x)

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a training step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        return self.compute_loss(batch, split="train")

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a validation step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        return self.compute_loss(batch, split="val")

    def test_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Execute a test step.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Test loss.
        """
        return self.compute_loss(batch, split="test")

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler.

        Returns
        -------
        Any
            PyTorch Lightning optimizer configuration.
        """
        parameters = self.trainable_parameters
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(parameters)
        else:
            optimizer = torch.optim.Adam(parameters)

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val/loss"}}
        return {"optimizer": optimizer}

    def predict(self, past: torch.Tensor, standard_name: str = "rainfall_rate") -> np.ndarray[Any, Any]:
        """Generate unnormalized forecasts from unnormalized past observations.

        Parameters
        ----------
        past : torch.Tensor
            Past observations with shape ``(T, H, W)``.
        standard_name : str, optional
            CF standard name that selects normalization and denormalization
            functions. Default is ``"rainfall_rate"``.

        Returns
        -------
        np.ndarray
            Forecast array shaped ``(ensemble_size, forecast_steps, H, W)`` for
            single-channel outputs.
        """
        if len(past.shape) != 3:
            raise ValueError("Input must be of shape (T, H, W)")

        past_clean = np.nan_to_num(past.cpu().numpy())
        past_clean = past_clean[np.newaxis, :, np.newaxis, ...]
        norm_func = NORMALIZATION_REGISTRY[standard_name]
        norm_past = norm_func(past_clean)

        x = torch.from_numpy(norm_past).to(self.device)
        self.eval()
        with torch.no_grad():
            preds_tensor = self.predict_normalized(x)

        preds_np: np.ndarray[Any, Any] = preds_tensor.cpu().numpy()
        denorm_func = DENORMALIZATION_REGISTRY[standard_name]
        preds_np = denorm_func(preds_np)
        preds_np = preds_np.squeeze(0)
        preds_np = np.swapaxes(preds_np, 0, 1)
        return preds_np


class OutputSpaceForecastingTaskModule(BaseForecastingTaskModule):
    """Lightning task module for direct forecasting in output space.

    Purpose
    -------
    This task module trains conventional forecasting models whose outputs can be
    compared directly against forecast targets in the original normalized data
    space. It is the generic wrapper used for models such as ConvGRU, where a
    single forward pass produces forecast tensors that are supervised directly.

    Ownership
    ---------
    This class owns:

    - the forecasting network passed in as ``network``
    - the output-space forecasting loss
    - optional masked-loss behavior using ``target_mask``
    - image and ensemble-statistic logging specific to direct forecast outputs

    It does not own:

    - source-data normalization rules outside the inherited ``predict`` helper
    - latent-space encoding or decoding components
    - sampler-driven generative forecast logic

    Training behavior
    -----------------
    A forecasting batch provides ``input`` and ``target`` tensors, plus an
    optional ``target_mask``. The module calls ``network(input)`` to obtain a
    normalized forecast tensor, optionally applies masked loss, and compares the
    network outputs directly against the target tensor in output space.

    Inference behavior
    ------------------
    Inference is a direct forward pass through the forecasting network. The
    inherited :meth:`predict` helper normalizes raw inputs, calls
    :meth:`predict_normalized`, and denormalizes the resulting forecast back to
    physical units.

    Parameters
    ----------
    network : torch.nn.Module
        Forecasting network to train.
    loss_class : type[torch.nn.Module] or str, optional
        Loss function class or registry name. Default is ``"mse"``.
    loss_params : dict or None, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    masked_loss : bool, optional
        Whether to use masked-loss computation with ``target_mask`` from the
        batch. Default is ``False``.
    optimizer : Callable[..., torch.optim.Optimizer] or None, optional
        Optimizer factory. Default is ``None`` (Adam).
    lr_scheduler : Callable[..., torch.optim.lr_scheduler.LRScheduler] or None, optional
        Learning-rate scheduler factory. Default is ``None``.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_class: type[torch.nn.Module] | str = "mse",
        loss_params: dict[str, Any] | None = None,
        masked_loss: bool = False,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        self.save_hyperparameters("loss_class", "loss_params", "masked_loss")
        self.network = network
        self.criterion = build_loss(loss_class=loss_class, loss_params=loss_params, masked_loss=masked_loss)
        self.log_images_iterations = [50, 100, 200, 500, 750, 1000, 2000, 5000]

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
    ) -> Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]:
        """Run the forecasting network.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Normalized input history tensor.

        Returns
        -------
        Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]
            Normalized forecast tensor with an explicit ensemble dimension.
        """
        return self.network(x)

    def compute_loss(self, batch: dict[str, torch.Tensor], split: str = "train") -> torch.Tensor:
        """Compute and log forecasting loss for one batch.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch containing ``input`` and ``target`` tensors, and
            optionally ``target_mask`` when masked loss is enabled.
        split : str, optional
            Current data split. Default is ``"train"``.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        past = batch["input"]
        future = batch["target"]
        preds = self(past).clamp(min=-1, max=1)

        # Flatten ensemble and channel dims for loss functions that expect
        # (B, T, M*C, H, W), preserving backward compatibility with CRPS etc.
        preds_flat = rearrange(preds, "b t m c h w -> b t (m c) h w")

        ensemble_size = getattr(self.network, "ensemble_size", 1)
        ensemble_std = preds.std(dim=2).mean() if ensemble_size > 1 else None

        if self.hparams["masked_loss"]:
            mask = batch["target_mask"]
            loss = self.criterion(preds_flat, future, mask)
        else:
            loss = self.criterion(preds_flat, future)

        if isinstance(loss, tuple):
            loss, log_dict = loss
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )

        self.log(f"{split}/loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)

        if ensemble_std is not None:
            self.log(f"{split}/ensemble_std", ensemble_std, on_epoch=True, sync_dist=True)

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
                logger=self.logger,  # type: ignore[arg-type]
                global_step=self.global_step,
                ensemble_size=ensemble_size,
                split=split,
            )
        return loss

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return the forecasting network parameters.

        Returns
        -------
        list of torch.nn.Parameter
            Parameters optimized for direct forecasting.
        """
        return list(self.network.parameters())

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "OutputSpaceForecastingTaskModule":
        """Load a forecasting task module from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the saved Lightning checkpoint.
        device : str, optional
            Device to map parameters onto. Default is ``"cpu"``.

        Returns
        -------
        OutputSpaceForecastingTaskModule
            Loaded output-space forecasting task module.
        """
        return cls.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(device),
            strict=True,
            weights_only=False,
        )


class LatentDiffusionTaskModule(BaseForecastingTaskModule):
    """Lightning task module for latent diffusion forecasting.

    Purpose
    -------
    This task module trains a latent diffusion forecasting system that reuses a
    stage-1 autoencoder. Forecast supervision is applied in latent space rather
    than directly on decoded forecast tensors. At inference time, the module
    samples forecast latents and decodes them back to the original data space.

    Ownership
    ---------
    This class owns:

    - the trained autoencoder reused from stage 1
    - the latent diffusion architecture
    - the latent diffusion loss
    - the diffusion sampler used for forecast generation
    - optional EMA tracking over diffusion-network weights

    It does not own:

    - stage-1 autoencoder training
    - output-space supervision for the diffusion loss
    - the source-data normalization rules beyond the inherited ``predict``
      helper

    Training behavior
    -----------------
    A forecasting batch provides raw normalized ``input`` and ``target``
    tensors. The reused autoencoder encoder maps both into latent space under
    ``torch.no_grad()``. The module then computes a diffusion loss entirely on
    latent tensors and exposes only the diffusion-network parameters through
    :attr:`trainable_parameters`, so the reused autoencoder remains frozen.

    Inference behavior
    ------------------
    Inference encodes the input history with the reused autoencoder, samples a
    latent forecast with the diffusion sampler, then decodes the sampled latent
    forecast back to data space. Ensemble generation is explicit here: the
    module repeats encoded inputs per requested ensemble member, samples a
    forecast latent for each member, and concatenates the decoded members along
    the channel dimension.

    Parameters
    ----------
    autoencoder : AutoencoderNet
        Trained autoencoder reused from stage 1. The encoder is used during
        stage-2 training to map forecasting inputs and targets into latent
        space. The decoder is retained for forecast inference but is not used in
        the stage-2 diffusion loss.
    diffusion_net : LatentDiffusionNet
        Latent diffusion architecture to train.
    forecast_steps : int
        Number of forecast timesteps decoded during inference.
    ensemble_size : int, optional
        Number of ensemble members decoded during inference. Default is ``1``.
    loss : DiffusionLoss or None, optional
        Latent diffusion loss module. If ``None``, ``DiffusionLoss`` is built
        from ``diffusion_net``. Default is ``None``.
    optimizer : Callable[..., torch.optim.Optimizer] or None, optional
        Optimizer factory. Default is ``None`` (Adam).
    lr_scheduler : Callable[..., torch.optim.lr_scheduler.LRScheduler] or None, optional
        Learning-rate scheduler factory. Default is ``None``.
    ema_decay : float or None, optional
        If provided, track an exponential moving average of diffusion-net
        parameters with this decay (commonly ``0.999`` or ``0.9999``). EMA-
        smoothed weights are swapped in during validation, testing, and
        prediction — the raw weights receive gradient updates during training.
        This is standard practice in diffusion models: the iterative denoising
        process amplifies small weight fluctuations, and EMA averaging
        suppresses that noise for cleaner samples at inference time. Default is
        ``None`` (no EMA).
    """

    def __init__(
        self,
        autoencoder: AutoencoderNet,
        diffusion_net: LatentDiffusionNet,
        forecast_steps: int,
        ensemble_size: int = 1,
        loss: DiffusionLoss | None = None,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        ema_decay: float | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, lr_scheduler=lr_scheduler)
        if forecast_steps < 1:
            raise ValueError(f"forecast_steps ({forecast_steps}) must be at least 1.")
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size ({ensemble_size}) must be at least 1.")

        self.save_hyperparameters("forecast_steps", "ensemble_size", "ema_decay")
        self.autoencoder = autoencoder
        self.diffusion_net = diffusion_net
        self.loss_fn = loss if loss is not None else DiffusionLoss(diffusion_net)
        self.sampler = DiffusionSampler(diffusion_net)
        self.ema = ExponentialMovingAverage(diffusion_net, decay=ema_decay) if ema_decay is not None else None

    def _freeze_autoencoder(self) -> None:
        """Freeze the reused autoencoder before stage-2 use.

        The same autoencoder instance is shared with stage-1 reconstruction
        training, so freezing must happen when the diffusion stage begins rather
        than in ``__init__``.
        """
        self.autoencoder.eval()
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = False

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch input_steps channels height width"],
    ) -> Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]:
        """Generate decoded forecasts from normalized input histories.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch input_steps channels height width"]
            Normalized input history tensor.

        Returns
        -------
        Float[torch.Tensor, "batch forecast_steps ensemble_size out_channels height width"]
            Decoded normalized forecast tensor with an explicit ensemble
            dimension.
        """
        input_latents = self.autoencoder.encode(x)
        repeated_input_latents = input_latents.repeat_interleave(self.hparams["ensemble_size"], dim=0)
        latent_shape = (
            x.shape[0] * self.hparams["ensemble_size"],
            input_latents.shape[1],
            self.hparams["forecast_steps"],
            input_latents.shape[3],
            input_latents.shape[4],
        )
        forecast_latents = self.sampler.sample(repeated_input_latents, latent_shape)
        decoded = self.autoencoder.decode(forecast_latents)
        # Decoded latent has shape (B*E, T, C, H, W) because ensemble members
        # were stacked in the batch dim via repeat_interleave.  Unstack into an
        # explicit ensemble dim and move time before ensemble for the standard
        # (B, T, E, C, H, W) shape contract expected by loss functions etc.
        return rearrange(decoded, "(b e) t c h w -> b t e c h w", e=self.hparams["ensemble_size"])

    def compute_loss(self, batch: dict[str, torch.Tensor], split: str = "train") -> torch.Tensor:
        """Compute latent diffusion loss for a forecasting batch.

        Parameters
        ----------
        batch : dict of str to torch.Tensor
            Forecasting batch containing ``input`` and ``target`` tensors.
        split : str, optional
            Current data split. Default is ``"train"``.

        Returns
        -------
        torch.Tensor
            Scalar latent diffusion loss.
        """
        with torch.no_grad():
            input_latents = self.autoencoder.encode(batch["input"])
            target_latents = self.autoencoder.encode(batch["target"])
        loss = self.loss_fn(input_latents, target_latents)
        self.log(f"{split}/loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)
        return loss

    @property
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return only the diffusion-network parameters.

        Returns
        -------
        list of torch.nn.Parameter
            Parameters optimized during stage-2 latent diffusion training.
        """
        return list(self.diffusion_net.parameters())

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update EMA after each training batch when enabled.

        Parameters
        ----------
        outputs : Any
            Lightning training outputs.
        batch : Any
            Batch passed to the training step.
        batch_idx : int
            Batch index supplied by Lightning.
        """
        del outputs, batch, batch_idx
        if self.ema is not None:
            self.ema.update()

    def on_fit_start(self) -> None:
        """Freeze the reused autoencoder before diffusion training starts."""
        self._freeze_autoencoder()

    def on_validation_start(self) -> None:
        """Swap EMA weights in before validation when enabled."""
        self._freeze_autoencoder()
        if self.ema is not None:
            self.ema.apply()

    def on_validation_end(self) -> None:
        """Restore raw diffusion weights after validation when enabled."""
        if self.ema is not None:
            self.ema.restore()

    def on_test_start(self) -> None:
        """Swap EMA weights in before testing when enabled."""
        self._freeze_autoencoder()
        if self.ema is not None:
            self.ema.apply()

    def on_test_end(self) -> None:
        """Restore raw diffusion weights after testing when enabled."""
        if self.ema is not None:
            self.ema.restore()

    def on_predict_start(self) -> None:
        """Swap EMA weights in before prediction when enabled."""
        self._freeze_autoencoder()
        if self.ema is not None:
            self.ema.apply()

    def on_predict_end(self) -> None:
        """Restore raw diffusion weights after prediction when enabled."""
        if self.ema is not None:
            self.ema.restore()

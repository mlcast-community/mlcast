"""Lightning module wrappers for reconstruction tasks."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.losses import build_loss


class ReconstructionTaskModule(pl.LightningModule):
    """Lightning task module for reconstruction training.

    Purpose
    -------
    This task module trains reconstruction models on tensor-only batches from
    ``ReconstructionDataset``. It is intended for stage-1 reconstruction or
    autoencoder training, where the model learns to reproduce normalized
    sequence windows.

    Ownership
    ---------
    This class owns:

    - the reconstruction network
    - the reconstruction loss defined by ``loss_class`` and ``loss_params``
    - the optimizer and learning-rate scheduler factories

    It does not own:

    - source-data normalization rules
    - forecasting-specific targets, masks, or ensemble behavior
    - latent diffusion training or sampler-driven inference logic

    Training behavior
    -----------------
    Each batch is a tensor-only reconstruction sample. The module uses that
    tensor as both the model input and the reconstruction target, computes the
    reconstruction loss directly in output space, and logs the resulting scalar
    loss for the active split.

    Inference behavior
    ------------------
    ``forward`` applies the reconstruction network to a normalized input tensor
    and returns a reconstructed normalized tensor of the same shape. This
    module does not implement forecasting-specific prediction helpers or any
    sampler-based inference path.

    Parameters
    ----------
    network : torch.nn.Module
        Reconstruction model that maps an input tensor back to the same shape.
    loss_class : type[torch.nn.Module] or str, optional
        Loss function class or registry name. Default is ``"mse"``.
    loss_params : dict or None, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    optimizer : Callable[..., torch.optim.Optimizer] or None, optional
        Optimizer factory used by :meth:`configure_optimizers`. Default is
        ``None`` (Adam over ``self.parameters()``).
    lr_scheduler : Callable[..., torch.optim.lr_scheduler.LRScheduler] or None, optional
        Learning-rate scheduler factory used by :meth:`configure_optimizers`.
        Default is ``None``.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_class: type[torch.nn.Module] | str = "mse",
        loss_params: dict[str, Any] | None = None,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("loss_class", "loss_params")
        self.network = network
        self.optimizer_factory = optimizer
        self.lr_scheduler_factory = lr_scheduler
        self.criterion = build_loss(loss_class=loss_class, loss_params=loss_params, masked_loss=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels height width"],
    ) -> Float[torch.Tensor, "batch time channels height width"]:
        """Run the reconstruction network.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Normalized reconstruction input.

        Returns
        -------
        Float[torch.Tensor, "batch time channels height width"]
            Reconstructed normalized tensor.
        """
        return self.network(x)

    def shared_step(self, batch: torch.Tensor, split: str = "train") -> torch.Tensor:
        """Compute reconstruction loss for one batch.

        Parameters
        ----------
        batch : torch.Tensor
            Tensor-only reconstruction batch.
        split : str, optional
            Current data split. Default is ``"train"``.

        Returns
        -------
        torch.Tensor
            Scalar reconstruction loss.
        """
        preds = self(batch).clamp(min=-1, max=1)
        loss = self.criterion(preds, batch)
        if isinstance(loss, tuple):
            loss, log_dict = loss
            self.log_dict(
                log_dict, prog_bar=False, logger=True, on_step=(split == "train"), on_epoch=True, sync_dist=True
            )
        self.log(f"{split}/rec_loss", loss, prog_bar=True, on_epoch=True, on_step=(split == "train"), sync_dist=True)
        return loss

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Execute a training step.

        Parameters
        ----------
        batch : torch.Tensor
            Reconstruction batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        return self.shared_step(batch, split="train")

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Execute a validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Reconstruction batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        return self.shared_step(batch, split="val")

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Execute a test step.

        Parameters
        ----------
        batch : torch.Tensor
            Reconstruction batch.
        _batch_idx : int
            Batch index supplied by Lightning.

        Returns
        -------
        torch.Tensor
            Test loss.
        """
        return self.shared_step(batch, split="test")

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler.

        Returns
        -------
        Any
            PyTorch Lightning optimizer configuration.
        """
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.Adam(self.parameters())

        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val/rec_loss"}}
        return {"optimizer": optimizer}

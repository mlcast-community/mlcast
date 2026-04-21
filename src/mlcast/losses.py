"""Loss functions for nowcasting models.

Provides CRPS, afCRPS, MaskedLoss, and a factory function to build
loss instances by name.
"""

from typing import Any

import torch
from torch import nn


class LossWithReduction(nn.Module):
    """Base class for losses with reduction options.

    Parameters
    ----------
    reduction : str, optional
        Reduction mode to apply to the loss. Must be one of ``'mean'``,
        ``'sum'``, or ``'none'``. Default is ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"], "reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction

    def apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply the specified reduction to the loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MaskedLoss(LossWithReduction):
    """Wrapper to apply a mask to a given loss function.

    Masks out invalid pixels before computing the loss, ensuring that only
    valid regions contribute to the final value.

    Parameters
    ----------
    elementwise_loss : nn.Module
        Base loss function to be masked. Must accept ``(preds, target)`` and
        return element-wise (unreduced) loss. Should be instantiated with
        ``reduction='none'``.
    reduction : str, optional
        Reduction mode applied after masking. Default is ``'mean'``.
    """

    def __init__(self, elementwise_loss: nn.Module, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.elementwise_loss = elementwise_loss

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions of shape (B, T, C, *D).
        target : torch.Tensor
            Target of shape (B, T, C, *D).
        mask : torch.Tensor
            Mask with 1 for valid and 0 for invalid pixels.
            Broadcasted to match preds/target shape if needed.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        elementwise_loss = self.elementwise_loss(preds, target)
        masked_loss = elementwise_loss * mask

        broadcast_factor = elementwise_loss.numel() // mask.numel()
        valid_pixels = mask.sum() * broadcast_factor
        if valid_pixels > 0:
            if self.reduction == "mean":
                return masked_loss.sum() / valid_pixels
            elif self.reduction == "sum":
                return masked_loss.sum()
            else:
                return masked_loss
        else:
            return torch.tensor(0.0, device=preds.device)


class CRPS(LossWithReduction):
    r"""Continuous Ranked Probability Score (CRPS) loss with optional temporal
    consistency regularization.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|], where X, X' are independent
    samples from the forecast distribution and y is the observation.

    Parameters
    ----------
    temporal_lambda : float, optional
        Weight for the temporal consistency penalty. Default is ``0.0``.
    reduction : str, optional
        Reduction mode. Default is ``'mean'``.

    Expected shapes
    ---------------
    preds : (B, T, M, \*D)
        Ensemble predictions with ensemble size M on dim=2.
    target : (B, T, C, \*D)
        Deterministic target with channel C on dim=2 (should be 1).
    """

    def __init__(self, temporal_lambda: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.temporal_lambda = temporal_lambda

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CRPS loss."""
        diff_to_target = torch.abs(preds - target)
        term1 = diff_to_target.mean(dim=2)

        preds_i = preds.unsqueeze(3)
        preds_j = preds.unsqueeze(2)
        pairwise_diff = torch.abs(preds_i - preds_j)
        term2 = 0.5 * pairwise_diff.mean(dim=(2, 3))

        crps = term1 - term2

        if self.temporal_lambda > 0:
            crps = crps.mean(dim=1)
            temporal_diff = preds[:, 1:, :, ...] - preds[:, :-1, :, ...]
            temporal_penalty = torch.abs(temporal_diff).mean(dim=(1, 2))
            crps = crps + self.temporal_lambda * temporal_penalty
            crps = crps[:, None, None, ...]
        else:
            crps = crps.unsqueeze(2)

        return self.apply_reduction(crps)


class AFCRPS(LossWithReduction):
    r"""Almost fair CRPS (afCRPS) loss as in eq. (4) of Lang et al. (2024).

    Interpolates between the standard CRPS and the fair CRPS via the
    fairness parameter ``alpha``.

    Parameters
    ----------
    alpha : float, optional
        Fairness parameter in ``(0, 1]``. Default is ``0.95``.
    temporal_lambda : float, optional
        Weight for the temporal consistency penalty. Default is ``0.0``.
    reduction : str, optional
        Reduction mode. Default is ``'mean'``.

    Expected shapes
    ---------------
    preds : (B, T, M, \*D)
        Ensemble predictions with ensemble size M on dim=2.
    target : (B, T, C, \*D)
        Deterministic target with channel C on dim=2 (should be 1).
    """

    def __init__(self, alpha: float = 0.95, temporal_lambda: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.temporal_lambda = temporal_lambda

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute afCRPS over an ensemble."""
        if preds.dim() < 3:
            raise ValueError("preds must have at least 3 dimensions.")
        if target.shape[0] != preds.shape[0]:
            raise ValueError("batch dimension of preds and target must match.")
        if target.shape[1] != preds.shape[1]:
            raise ValueError("time dimension of preds and target must match.")

        M = preds.shape[2]
        if M < 2:
            raise ValueError("Ensemble size M must be >= 2 for afCRPS.")

        eps = (1.0 - self.alpha) / float(M)

        abs_x_minus_y = (preds - target).abs()

        x_j = preds.unsqueeze(3)
        x_k = preds.unsqueeze(2)

        abs_xj_minus_y = abs_x_minus_y.unsqueeze(3)
        abs_xk_minus_y = abs_x_minus_y.unsqueeze(2)
        abs_xj_minus_xk = (x_j - x_k).abs()

        term = abs_xj_minus_y + abs_xk_minus_y - (1.0 - eps) * abs_xj_minus_xk

        idx = torch.arange(M, device=preds.device)
        mask = idx[:, None] != idx[None, :]
        term = term * mask.view(1, 1, M, M, *([1] * (term.dim() - 4)))

        summed = term.sum(dim=(2, 3))
        afcrps = summed / (2.0 * M * (M - 1))

        if self.temporal_lambda > 0:
            afcrps = afcrps.mean(dim=1)
            temporal_diff = preds[:, 1:, :, ...] - preds[:, :-1, :, ...]
            temporal_penalty = torch.abs(temporal_diff).mean(dim=(1, 2))
            afcrps = afcrps + self.temporal_lambda * temporal_penalty
            afcrps = afcrps[:, None, None, ...]
        else:
            afcrps = afcrps.unsqueeze(2)

        return self.apply_reduction(afcrps)


PIXEL_LOSSES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "crps": CRPS, "afcrps": AFCRPS}


def build_loss(
    loss_class: type[nn.Module] | str = "mse",
    loss_params: dict[str, Any] | None = None,
    masked_loss: bool = False,
) -> nn.Module:
    """Build a loss function, optionally wrapped with masking.

    Parameters
    ----------
    loss_class : type[nn.Module] or str, optional
        Loss class or its string name. Accepted names: ``'mse'``, ``'mae'``,
        ``'crps'``, ``'afcrps'``. If a class is provided, it must be a callable
        (e.g., a subclass of nn.Module) that accepts the keyword arguments
        provided in ``loss_params``. Default is ``'mse'``.
    loss_params : dict, optional
        Keyword arguments for the loss constructor. Default is ``None``.
    masked_loss : bool, optional
        If ``True``, wrap in :class:`MaskedLoss`. Default is ``False``.

    Returns
    -------
    criterion : nn.Module
        Instantiated loss module.
    """
    if not isinstance(loss_class, str | type):
        raise TypeError(f"loss_class must be a string or a class, got {type(loss_class)}")

    if isinstance(loss_class, str):
        if loss_class.lower() not in PIXEL_LOSSES:
            raise ValueError(f"Unknown loss class '{loss_class}'. Available: {list(PIXEL_LOSSES.keys())}")
        LossClass = PIXEL_LOSSES[loss_class.lower()]
    else:
        LossClass = loss_class

    params = loss_params.copy() if loss_params is not None else None

    if masked_loss and params is not None:
        reduction = params.pop("reduction", "mean")
        criterion = MaskedLoss(LossClass(reduction="none", **params), reduction=reduction)
    elif masked_loss:
        criterion = MaskedLoss(LossClass(reduction="none"), reduction="mean")
    else:
        if params is not None:
            criterion = LossClass(**params)
        else:
            criterion = LossClass()

    return criterion

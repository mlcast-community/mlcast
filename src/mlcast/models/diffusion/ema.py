"""Exponential moving average helpers for diffusion weights."""

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Track an exponential moving average of trainable module parameters.

    Parameters
    ----------
    module : nn.Module
        Module whose parameters should be tracked.
    decay : float, optional
        EMA decay factor. Default is ``0.999``.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay ({decay}) must be in [0, 1).")
        self.module = module
        self.decay = decay
        self.shadow_params = [
            parameter.detach().clone() for parameter in module.parameters() if parameter.requires_grad
        ]
        self.backup_params: list[torch.Tensor] | None = None

    def update(self) -> None:
        """Update EMA shadow parameters from the current module parameters."""
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        for i, (shadow_param, parameter) in enumerate(zip(self.shadow_params, trainable_params, strict=True)):
            if shadow_param.device != parameter.device:
                self.shadow_params[i] = shadow_param.to(parameter.device)
                shadow_param = self.shadow_params[i]
            shadow_param.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def _align_device(self) -> None:
        """Move shadow parameters to the current device of the module's parameters."""
        for i, shadow_param in enumerate(self.shadow_params):
            ref_param = next(self.module.parameters())
            if shadow_param.device != ref_param.device:
                self.shadow_params[i] = shadow_param.to(ref_param.device)

    def apply(self) -> None:
        """Swap EMA parameters into the tracked module."""
        self._align_device()
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        self.backup_params = [parameter.detach().clone() for parameter in trainable_params]
        for parameter, shadow_param in zip(trainable_params, self.shadow_params, strict=True):
            parameter.data.copy_(shadow_param.data)

    def restore(self) -> None:
        """Restore module parameters saved before :meth:`apply`."""
        if self.backup_params is None:
            raise RuntimeError("EMA restore() called before apply().")
        trainable_params = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        for parameter, backup_param in zip(trainable_params, self.backup_params, strict=True):
            if backup_param.device != parameter.device:
                backup_param = backup_param.to(parameter.device)
            parameter.data.copy_(backup_param.data)
        self.backup_params = None

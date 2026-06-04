"""Diffusion noise schedules."""

import torch


class DiffusionScheduler:
    """Linear-beta diffusion scheduler.

    Parameters
    ----------
    timesteps : int, optional
        Number of diffusion timesteps. Default is ``100``.
    beta_start : float, optional
        Initial beta value. Default is ``1e-4``.
    beta_end : float, optional
        Final beta value. Default is ``2e-2``.
    """

    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        if timesteps < 1:
            raise ValueError(f"timesteps ({timesteps}) must be at least 1.")
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def buffers(self, device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """Build schedule tensors for registration as module buffers.

        Parameters
        ----------
        device : torch.device
            Device on which buffers should be allocated.
        dtype : torch.dtype, optional
            Floating-point dtype for schedule tensors. Default is
            ``torch.float32``.

        Returns
        -------
        dict of str to torch.Tensor
            Schedule tensors used for forward and reverse diffusion.
        """
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=device, dtype=dtype)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }


def extract_schedule_value(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Gather schedule values and reshape them for broadcasting.

    Parameters
    ----------
    values : Float[torch.Tensor, "timesteps"]
        One-dimensional schedule tensor.
    timesteps : Int[torch.Tensor, "batch"]
        Timestep index for each batch item.
    target_shape : torch.Size
        Shape of the target tensor the values should broadcast against.

    Returns
    -------
    torch.Tensor
        Gathered values reshaped to ``(batch, 1, ..., 1)``.
    """
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *([1] * (len(target_shape) - 1)))

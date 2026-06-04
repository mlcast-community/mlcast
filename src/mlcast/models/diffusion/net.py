"""Latent diffusion network composed from conditioner and denoiser modules."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.models.diffusion.conditioner import ConditionerNet
from mlcast.models.diffusion.denoiser import DenoiserUNet
from mlcast.models.diffusion.scheduler import DiffusionScheduler, extract_schedule_value


class LatentDiffusionNet(nn.Module):
    """Trainable latent diffusion denoising network.

    Parameters
    ----------
    conditioner : ConditionerNet
        Network that builds context from input-history latents. It must accept
        ``Float[Tensor, "batch latent_channels input_time height width"]`` and
        return ``Float[Tensor, "batch condition_channels input_time height width"]``.
    denoiser : DenoiserUNet
        Network that predicts noise from noised target latents. It must accept
        ``noisy`` with shape
        ``Float[Tensor, "batch latent_channels forecast_time height width"]``,
        ``timesteps`` with shape ``(batch,)``, and ``context`` with shape
        ``Float[Tensor, "batch condition_channels input_time height width"]``;
        it must return
        ``Float[Tensor, "batch latent_channels forecast_time height width"]``.
    scheduler : DiffusionScheduler
        Diffusion noise scheduler. Calling ``scheduler.buffers(device, dtype)``
        must return one-dimensional tensors of length ``scheduler.timesteps``
        for ``sqrt_alphas_cumprod`` and ``sqrt_one_minus_alphas_cumprod`` so
        they can be gathered with timestep indices shaped ``(batch,)`` and
        broadcast over latent tensors shaped
        ``(batch, latent_channels, forecast_time, height, width)``.
    """

    def __init__(self, conditioner: ConditionerNet, denoiser: DenoiserUNet, scheduler: DiffusionScheduler) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.num_timesteps = scheduler.timesteps
        for name, value in scheduler.buffers(device=torch.device("cpu")).items():
            self.register_buffer(name, value)

    @jaxtyped(typechecker=beartype)
    def q_sample(
        self,
        x0: Float[torch.Tensor, "batch channels time height width"],
        timesteps: torch.Tensor,
        noise: Float[torch.Tensor, "batch channels time height width"] | None = None,
    ) -> Float[torch.Tensor, "batch channels time height width"]:
        """Diffuse clean latents to a chosen timestep.

        Parameters
        ----------
        x0 : Float[torch.Tensor, "batch channels time height width"]
            Clean target latent.
        timesteps : torch.Tensor
            Diffusion timestep for each sample.
        noise : Float[torch.Tensor, "batch channels time height width"] or None, optional
            Noise to add. If ``None``, standard Gaussian noise is sampled.
            Default is ``None``.

        Returns
        -------
        Float[torch.Tensor, "batch channels time height width"]
            Noised target latent.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = extract_schedule_value(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus_alpha = extract_schedule_value(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noised_target: Float[torch.Tensor, "batch latent_channels forecast_time height width"],
        timesteps: torch.Tensor,
        input_latents: Float[torch.Tensor, "batch latent_channels input_time height width"],
    ) -> Float[torch.Tensor, "batch latent_channels forecast_time height width"]:
        """Predict noise from a noised target latent and input context.

        Parameters
        ----------
        noised_target : Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Noised target latent.
        timesteps : torch.Tensor
            Diffusion timestep for each sample.
        input_latents : Float[torch.Tensor, "batch latent_channels input_time height width"]
            Encoded input-history latents.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Predicted noise.
        """
        context = self.conditioner(input_latents)
        return self.denoiser(noised_target, timesteps, context=context)

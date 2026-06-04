"""Simple ancestral sampler for latent diffusion models."""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.models.diffusion.net import LatentDiffusionNet
from mlcast.models.diffusion.scheduler import extract_schedule_value


class DiffusionSampler:
    """Generate latent samples with a compact DDPM-style reverse process.

    Parameters
    ----------
    net : LatentDiffusionNet
        Trained diffusion network.
    """

    def __init__(self, net: LatentDiffusionNet) -> None:
        self.net = net

    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        input_latents: Float[torch.Tensor, "batch latent_channels input_time height width"],
        output_shape: tuple[int, int, int, int, int],
    ) -> Float[torch.Tensor, "batch latent_channels forecast_time height width"]:
        """Sample forecast latents conditioned on input latents.

        Parameters
        ----------
        input_latents : Float[torch.Tensor, "batch latent_channels input_time height width"]
            Encoded input-history latents.
        output_shape : tuple of int
            Shape of the forecast latent to sample, ordered as
            ``(batch, channels, forecast_time, height, width)``.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Sampled forecast latent.
        """
        x = torch.randn(output_shape, device=input_latents.device, dtype=input_latents.dtype)
        for step in reversed(range(self.net.num_timesteps)):
            timesteps = torch.full((output_shape[0],), step, device=input_latents.device, dtype=torch.long)
            predicted_noise = self.net(x, timesteps, input_latents)
            sqrt_alpha = extract_schedule_value(self.net.sqrt_alphas_cumprod, timesteps, x.shape)
            sqrt_one_minus_alpha = extract_schedule_value(self.net.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
            x0 = (x - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha.clamp_min(1e-6)
            if step > 0:
                prev_timesteps = timesteps - 1
                prev_sqrt_alpha = extract_schedule_value(self.net.sqrt_alphas_cumprod, prev_timesteps, x.shape)
                prev_sqrt_one_minus_alpha = extract_schedule_value(
                    self.net.sqrt_one_minus_alphas_cumprod, prev_timesteps, x.shape
                )
                x = prev_sqrt_alpha * x0 + prev_sqrt_one_minus_alpha * predicted_noise
            else:
                x = x0
        return x

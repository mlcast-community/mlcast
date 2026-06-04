"""Loss helpers for latent diffusion training."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.models.diffusion.net import LatentDiffusionNet


class DiffusionLoss(nn.Module):
    """Noise-prediction MSE loss for latent diffusion.

    Parameters
    ----------
    net : LatentDiffusionNet
        Diffusion network used to sample noised latents and predict noise.
    """

    def __init__(self, net: LatentDiffusionNet) -> None:
        super().__init__()
        self.net = net

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        input_latents: Float[torch.Tensor, "batch latent_channels input_time height width"],
        target_latents: Float[torch.Tensor, "batch latent_channels forecast_time height width"],
    ) -> torch.Tensor:
        """Compute a random-timestep noise-prediction loss.

        Parameters
        ----------
        input_latents : Float[torch.Tensor, "batch latent_channels input_time height width"]
            Encoded input-history latents used as conditioning.
        target_latents : Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Clean target latents to diffuse.

        Returns
        -------
        torch.Tensor
            Scalar mean squared error between predicted and sampled noise.
        """
        timesteps = torch.randint(0, self.net.num_timesteps, (target_latents.shape[0],), device=target_latents.device)
        noise = torch.randn_like(target_latents)
        noised_target = self.net.q_sample(target_latents, timesteps=timesteps, noise=noise)
        predicted_noise = self.net(noised_target, timesteps, input_latents)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

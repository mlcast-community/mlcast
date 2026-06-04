"""Latent conditioning blocks for diffusion forecasting."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class ConditionerBlock(nn.Module):
    """Residual 3D-convolution block for latent conditioning.

    Parameters
    ----------
    channels : int
        Number of latent conditioning channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch channels time height width"]:
        """Apply residual conditioning refinement.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch channels time height width"]
            Latent conditioning tensor.

        Returns
        -------
        Float[torch.Tensor, "batch channels time height width"]
            Refined conditioning tensor.
        """
        return x + self.net(x)


class ConditionerNet(nn.Module):
    """Condition latent target denoising on encoded input history.

    Parameters
    ----------
    latent_channels : int
        Number of latent channels in the encoded input history.
    hidden_channels : int, optional
        Number of channels emitted as conditioning context. Default is ``32``.
    num_blocks : int, optional
        Number of residual conditioning blocks. Default is ``2``.
    """

    def __init__(self, latent_channels: int, hidden_channels: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks ({num_blocks}) must be at least 1.")

        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.input_projection = nn.Conv3d(latent_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(*(ConditionerBlock(hidden_channels) for _ in range(num_blocks)))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, z: Float[torch.Tensor, "batch latent_channels input_time height width"]
    ) -> Float[torch.Tensor, "batch hidden_channels input_time height width"]:
        """Build conditioning context from input-history latents.

        Parameters
        ----------
        z : Float[torch.Tensor, "batch latent_channels input_time height width"]
            Encoded input-history latent tensor.

        Returns
        -------
        Float[torch.Tensor, "batch hidden_channels input_time height width"]
            Conditioning context for the denoiser.
        """
        return self.blocks(self.input_projection(z))

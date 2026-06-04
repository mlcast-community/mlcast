"""Timestep-aware denoising network for latent diffusion."""

import math

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP.

    Parameters
    ----------
    embedding_dim : int
        Number of channels in the generated timestep embedding.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, timesteps: torch.Tensor) -> Float[torch.Tensor, "batch embedding_dim"]:
        """Embed integer diffusion timesteps.

        Parameters
        ----------
        timesteps : torch.Tensor
            Integer diffusion timesteps.

        Returns
        -------
        Float[torch.Tensor, "batch embedding_dim"]
            Projected sinusoidal timestep embeddings.
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -(math.log(10_000.0) / max(half_dim - 1, 1))
        )
        args = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.projection(embedding)


class _DenoiserBlock(nn.Module):
    """Internal residual denoising block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    timestep_channels : int
        Number of channels in the timestep embedding.
    """

    def __init__(self, in_channels: int, out_channels: int, timestep_channels: int) -> None:
        super().__init__()
        self.timestep_projection = nn.Linear(timestep_channels, out_channels)
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        """Apply timestep-conditioned residual denoising.

        Parameters
        ----------
        x : torch.Tensor
            Hidden denoising tensor.
        timestep_embedding : torch.Tensor
            Timestep embedding for each batch item.

        Returns
        -------
        torch.Tensor
            Updated hidden tensor.
        """
        timestep_bias = self.timestep_projection(timestep_embedding)[:, :, None, None, None]
        h = self.net(x)
        return self.skip_connection(x) + h + timestep_bias


class _SpatialDownsample(nn.Module):
    """Halve latent spatial resolution while preserving time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=(1, 2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the spatial dimensions of a latent tensor.

        Parameters
        ----------
        x : torch.Tensor
            Channel-first latent tensor.

        Returns
        -------
        torch.Tensor
            Tensor with half spatial resolution.
        """
        return self.op(x)


class _SpatialUpsample(nn.Module):
    """Double latent spatial resolution while preserving time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the spatial dimensions of a latent tensor.

        Parameters
        ----------
        x : torch.Tensor
            Channel-first latent tensor.

        Returns
        -------
        torch.Tensor
            Tensor with doubled spatial resolution.
        """
        return self.op(x)


class DenoiserUNet(nn.Module):
    """Compact timestep-aware U-Net denoiser for latent tensors.

    This is a real U-Net-style architecture: it builds a spatial downsampling
    path, applies a bottleneck at the lowest spatial resolution, upsamples back
    to the original latent resolution, and concatenates matching-resolution
    skip connections from the down path into the up path. It differs from a
    plain image U-Net because it operates on 3D latent tensors and only changes
    spatial resolution; the temporal dimension is preserved throughout. Each
    residual block also receives a diffusion timestep embedding.

    Parameters
    ----------
    latent_channels : int
        Number of channels in the noisy target latent.
    condition_channels : int
        Number of channels emitted by the conditioner.
    hidden_channels : int, optional
        Number of hidden channels in the denoiser. Default is ``32``.
    num_blocks : int, optional
        Number of U-Net resolution levels. Default is ``2``.
    """

    def __init__(
        self,
        latent_channels: int,
        condition_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks ({num_blocks}) must be at least 1.")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.timestep_embedding = TimestepEmbedding(hidden_channels)
        self.input_projection = nn.Conv3d(latent_channels + condition_channels, hidden_channels, kernel_size=1)
        self.down_blocks = nn.ModuleList(
            _DenoiserBlock(hidden_channels, hidden_channels, hidden_channels) for _ in range(num_blocks)
        )
        self.downsamples = nn.ModuleList(_SpatialDownsample(hidden_channels) for _ in range(num_blocks - 1))
        self.bottleneck = _DenoiserBlock(hidden_channels, hidden_channels, hidden_channels)
        self.upsamples = nn.ModuleList(_SpatialUpsample(hidden_channels) for _ in range(num_blocks - 1))
        self.up_blocks = nn.ModuleList(
            _DenoiserBlock(hidden_channels * 2, hidden_channels, hidden_channels) for _ in range(num_blocks - 1)
        )
        self.output_projection = nn.Conv3d(hidden_channels, latent_channels, kernel_size=1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noisy: Float[torch.Tensor, "batch latent_channels forecast_time height width"],
        timesteps: torch.Tensor,
        context: Float[torch.Tensor, "batch condition_channels input_time height width"],
    ) -> Float[torch.Tensor, "batch latent_channels forecast_time height width"]:
        """Predict noise in a noised latent target.

        Parameters
        ----------
        noisy : Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Noised target latent.
        timesteps : torch.Tensor
            Diffusion timestep for each sample.
        context : Float[torch.Tensor, "batch condition_channels input_time height width"]
            Conditioning context from the input-history latent.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Predicted noise tensor.
        """
        if context.shape[2] != noisy.shape[2]:
            context = torch.nn.functional.interpolate(context, size=noisy.shape[2:], mode="nearest")

        x = self.input_projection(torch.cat([noisy, context], dim=1))
        timestep_embedding = self.timestep_embedding(timesteps)

        skips: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.down_blocks):
            x = block(x, timestep_embedding)
            if block_idx < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[block_idx](x)

        x = self.bottleneck(x, timestep_embedding)

        for upsample, block in zip(self.upsamples, self.up_blocks, strict=True):
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = block(torch.cat([x, skip], dim=1), timestep_embedding)

        return self.output_projection(x)

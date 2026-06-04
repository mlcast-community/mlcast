"""Encoder blocks for the reconstruction autoencoder."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class EncoderBlock(nn.Module):
    """Spatio-temporal encoder block with optional spatial downsampling.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the block.
    downsample : bool, optional
        If ``True``, halve the spatial resolution with a stride-2 convolution.
        Default is ``True``.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        spatial_stride = 2 if downsample else 1
        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(1, spatial_stride, spatial_stride),
                padding=1,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch out_channels time out_height out_width"]:
        """Encode a channel-first spatio-temporal tensor.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch channels time height width"]
            Input tensor.

        Returns
        -------
        Float[torch.Tensor, "batch out_channels time out_height out_width"]
            Encoded tensor.
        """
        return self.net(x)


class Encoder(nn.Module):
    """Convolutional encoder for sequence reconstruction.

    Parameters
    ----------
    input_channels : int
        Number of channels in the source data.
    hidden_channels : int, optional
        Number of channels used in the first encoder block. Default is ``16``.
    latent_channels : int, optional
        Number of channels in the latent representation. Default is ``32``.
    num_blocks : int, optional
        Number of spatial downsampling blocks. Default is ``2``.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        latent_channels: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks ({num_blocks}) must be at least 1.")

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks

        layers: list[nn.Module] = []
        in_channels = input_channels
        for block_idx in range(num_blocks):
            out_channels = latent_channels if block_idx == num_blocks - 1 else hidden_channels * 2**block_idx
            layers.append(EncoderBlock(in_channels=in_channels, out_channels=out_channels, downsample=True))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch time channels height width"]
    ) -> Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]:
        """Encode a time-first sequence tensor into a latent tensor.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input sequence in the data-layer tensor layout.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
            Latent tensor in channel-first 3D-convolution layout.
        """
        return self.blocks(x.movedim(2, 1))

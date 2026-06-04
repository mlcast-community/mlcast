"""Decoder blocks for the reconstruction autoencoder."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class DecoderBlock(nn.Module):
    """Spatio-temporal decoder block with optional spatial upsampling.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the block.
    upsample : bool, optional
        If ``True``, double the spatial resolution with a transposed
        convolution. Default is ``True``.
    """

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = True) -> None:
        super().__init__()
        spatial_stride = 2 if upsample else 1
        output_padding = (0, 1, 1) if upsample else 0
        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(1, spatial_stride, spatial_stride),
                padding=1,
                output_padding=output_padding,
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
        """Decode a channel-first spatio-temporal tensor.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch channels time height width"]
            Input tensor.

        Returns
        -------
        Float[torch.Tensor, "batch out_channels time out_height out_width"]
            Decoded tensor.
        """
        return self.net(x)


class Decoder(nn.Module):
    """Convolutional decoder for sequence reconstruction.

    Parameters
    ----------
    output_channels : int
        Number of channels in the reconstructed source data.
    hidden_channels : int, optional
        Number of channels used near the output side of the decoder. Default is
        ``16``.
    latent_channels : int, optional
        Number of channels in the latent representation. Default is ``32``.
    num_blocks : int, optional
        Number of spatial upsampling blocks. Default is ``2``.
    """

    def __init__(
        self,
        output_channels: int,
        hidden_channels: int = 16,
        latent_channels: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks ({num_blocks}) must be at least 1.")

        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.num_blocks = num_blocks

        layers: list[nn.Module] = []
        in_channels = latent_channels
        for block_idx in range(num_blocks):
            is_last = block_idx == num_blocks - 1
            remaining_blocks = num_blocks - block_idx - 2
            out_channels = output_channels if is_last else hidden_channels * 2 ** max(remaining_blocks, 0)
            layers.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels, upsample=True))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, z: Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
    ) -> Float[torch.Tensor, "batch time channels height width"]:
        """Decode a latent tensor into a time-first reconstruction tensor.

        Parameters
        ----------
        z : Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
            Latent tensor in channel-first 3D-convolution layout.

        Returns
        -------
        Float[torch.Tensor, "batch time channels height width"]
            Reconstructed sequence in the data-layer tensor layout.
        """
        return self.blocks(z).movedim(1, 2)

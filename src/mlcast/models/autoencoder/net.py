"""Autoencoder network for reconstruction pretraining."""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped

from mlcast.models.autoencoder.decoder import Decoder
from mlcast.models.autoencoder.encoder import Encoder


class AutoencoderNet(nn.Module):
    """Compose an encoder and decoder into a reconstruction network.

    Parameters
    ----------
    encoder : Encoder
        Encoder module that maps input sequences to latent tensors.
    decoder : Decoder
        Decoder module that maps latent tensors back to input-space sequences.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @jaxtyped(typechecker=beartype)
    def encode(
        self, x: Float[torch.Tensor, "batch time channels height width"]
    ) -> Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]:
        """Encode an input sequence into latent space.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input sequence tensor.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
            Latent tensor produced by the encoder.
        """
        return self.encoder(x)

    @jaxtyped(typechecker=beartype)
    def decode(
        self, z: Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
    ) -> Float[torch.Tensor, "batch time channels height width"]:
        """Decode a latent tensor into input space.

        Parameters
        ----------
        z : Float[torch.Tensor, "batch latent_channels time latent_height latent_width"]
            Latent tensor produced by the encoder.

        Returns
        -------
        Float[torch.Tensor, "batch time channels height width"]
            Reconstructed sequence tensor.
        """
        return self.decoder(z)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch time channels height width"]
    ) -> Float[torch.Tensor, "batch time channels height width"]:
        """Run an end-to-end reconstruction forward pass.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input sequence tensor.

        Returns
        -------
        Float[torch.Tensor, "batch time channels height width"]
            Reconstructed sequence tensor.
        """
        return self.decode(self.encode(x))

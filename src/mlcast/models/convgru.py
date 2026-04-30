"""ConvGRU encoder-decoder architecture for spatio-temporal forecasting.

Provides the building blocks (ConvGRUCell, ConvGRU, Encoder, Decoder) and
the full EncoderDecoder model with optional ensemble generation via noisy
decoder inputs.
"""

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on 2D spatial grids.

    Implements a single-step GRU update where all linear projections are
    replaced by 2D convolutions, preserving spatial structure.

    Parameters
    ----------
    input_size : int
        Number of channels in the input tensor.
    hidden_size : int
        Number of channels in the hidden state.
    kernel_size : int, optional
        Kernel size for the convolutional gates. Default is ``3``.
    conv_layer : nn.Module, optional
        Convolutional layer class to use. Default is ``nn.Conv2d``.
    """

    def __init__(
        self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: type[nn.Module] = nn.Conv2d
    ):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.combined_gates = conv_layer(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_layer(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        inpt: Float[torch.Tensor, "batch in_channels height width"] | None = None,
        h_s: Float[torch.Tensor, "batch hidden_channels height width"] | None = None,
    ) -> Float[torch.Tensor, "batch hidden_channels height width"]:
        """Forward the ConvGRU cell for a single timestep.

        Parameters
        ----------
        inpt : Float[torch.Tensor, "batch in_channels height width"] or None, optional
            Input tensor.
        h_s : Float[torch.Tensor, "batch hidden_channels height width"] or None, optional
            Hidden state tensor.

        Returns
        -------
        new_state : Float[torch.Tensor, "batch hidden_channels height width"]
            Updated hidden state.
        """
        if h_s is None and inpt is None:
            raise ValueError("Both input and state can't be None")
        elif h_s is None and inpt is not None:
            h_s = torch.zeros(
                inpt.size(0), self.hidden_size, inpt.size(2), inpt.size(3), dtype=inpt.dtype, device=inpt.device
            )
        elif inpt is None and h_s is not None:
            inpt = torch.zeros(
                h_s.size(0), self.input_size, h_s.size(2), h_s.size(3), dtype=h_s.dtype, device=h_s.device
            )

        assert inpt is not None
        assert h_s is not None

        gamma, beta = torch.chunk(self.combined_gates(torch.cat([inpt, h_s], dim=1)), 2, dim=1)
        update = torch.sigmoid(gamma)
        reset = torch.sigmoid(beta)

        out_inputs = torch.tanh(self.out_gate(torch.cat([inpt, h_s * reset], dim=1)))
        new_state = h_s * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):
    """Convolutional GRU that unrolls a :class:`ConvGRUCell` over a sequence.

    Parameters
    ----------
    input_size : int
        Number of channels in the input tensor.
    hidden_size : int
        Number of channels in the hidden state.
    kernel_size : int, optional
        Kernel size for the convolutional gates. Default is ``3``.
    conv_layer : nn.Module, optional
        Convolutional layer class to use. Default is ``nn.Conv2d``.
    """

    def __init__(
        self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: type[nn.Module] = nn.Conv2d
    ):
        super().__init__()
        self.cell = ConvGRUCell(input_size, hidden_size, kernel_size, conv_layer)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time in_channels height width"] | None = None,
        h: Float[torch.Tensor, "batch hidden_channels height width"] | None = None,
    ) -> Float[torch.Tensor, "batch time hidden_channels height width"]:
        """Unroll the ConvGRU cell over the sequence (time) dimension.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time in_channels height width"] or None, optional
            Input sequence.
        h : Float[torch.Tensor, "batch hidden_channels height width"] or None, optional
            Initial hidden state.

        Returns
        -------
        hidden_states : Float[torch.Tensor, "batch time hidden_channels height width"]
            Stacked hidden states.
        """
        if x is None:
            raise ValueError("Input sequence x cannot be None")

        h_s = []
        for i in range(x.size(1)):
            h = self.cell(x[:, i], h)
            h_s.append(h)
        return torch.stack(h_s, dim=1)


class EncoderBlock(nn.Module):
    """ConvGRU-based encoder block with spatial downsampling.

    Applies a :class:`ConvGRU` followed by ``nn.PixelUnshuffle(2)`` to
    halve spatial dimensions and quadruple channels.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    kernel_size : int, optional
        Kernel size for the ConvGRU. Default is ``3``.
    conv_layer : nn.Module, optional
        Convolutional layer class to use. Default is ``nn.Conv2d``.
    """

    def __init__(self, input_size: int, kernel_size: int = 3, conv_layer: type[nn.Module] = nn.Conv2d):
        super().__init__()
        self.convgru = ConvGRU(input_size, input_size, kernel_size, conv_layer)
        self.down = nn.PixelUnshuffle(2)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch time in_channels height width"]
    ) -> Float[torch.Tensor, "batch time out_channels out_height out_width"]:
        """Forward the encoder block.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time in_channels height width"]
            Input sequence.

        Returns
        -------
        out : Float[torch.Tensor, "batch time out_channels out_height out_width"]
            Downsampled tensor, where out_channels is 4*in_channels,
            and spatial dimensions are halved.
        """
        x = self.convgru(x)
        x = self.down(x)
        return x


class Encoder(nn.Module):
    r"""ConvGRU-based encoder that stacks multiple :class:`EncoderBlock` layers.

    After each block the spatial resolution is halved via pixel-unshuffle.

    .. code-block:: text

         ///    Encoder Block 1    \\\                ///    Encoder Block 2    \\\
     /--------------------------------------------\ /---------------------------------------\
    |                                              |                                         |
    *        *---------*      *-----------------*  *   *---------*      *-----------------*  *
        X -> | ConvGRU | ---> | Pixel Unshuffle | ---> | ConvGRU | ---> | Pixel Unshuffle | ---> ...
        |    *---------*  |   *-----------------*  |   *---------*  |   *-----------------*  |
        v                 v                        v                v                        v
      (b,t,c,h,w)      (b,t,c,h,w)          (b,t,c*4,h/2,w/2) (b,t,c*4,h/2,w/2)    (b,t,c*16,h/4,w/4)

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels. Default is ``1``.
    num_blocks : int, optional
        Number of encoder blocks to stack. Default is ``4``.
    **kwargs
        Additional keyword arguments forwarded to each :class:`EncoderBlock`.
    """

    def __init__(self, input_channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.channel_sizes = [input_channels * 4**i for i in range(num_blocks)]
        self.blocks = nn.ModuleList([EncoderBlock(self.channel_sizes[i], **kwargs) for i in range(num_blocks)])

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch time channels height width"]
    ) -> list[Float[torch.Tensor, "batch time _ _ _"]]:
        """Forward the encoder through all blocks.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input sequence.

        Returns
        -------
        hidden_states : list of Float[torch.Tensor, "batch time _ _ _"]
            Hidden state tensors from each block, with progressively reduced
            spatial dimensions.
        """
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class DecoderBlock(nn.Module):
    """ConvGRU-based decoder block with spatial upsampling.

    Applies a :class:`ConvGRU` followed by ``nn.PixelShuffle(2)`` to double
    spatial dimensions and quarter channels.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    hidden_size : int
        Number of hidden channels for the ConvGRU.
    kernel_size : int, optional
        Kernel size for the ConvGRU. Default is ``3``.
    conv_layer : nn.Module, optional
        Convolutional layer class to use. Default is ``nn.Conv2d``.
    """

    def __init__(
        self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: type[nn.Module] = nn.Conv2d
    ):
        super().__init__()
        self.convgru = ConvGRU(input_size, hidden_size, kernel_size, conv_layer)
        self.up = nn.PixelShuffle(2)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time in_channels height width"],
        hidden_state: Float[torch.Tensor, "batch in_channels height width"],
    ) -> Float[torch.Tensor, "batch time out_channels out_height out_width"]:
        """Forward the decoder block.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time in_channels height width"]
            Input sequence.
        hidden_state : Float[torch.Tensor, "batch in_channels height width"]
            Hidden state from the corresponding encoder block.

        Returns
        -------
        out : Float[torch.Tensor, "batch time out_channels out_height out_width"]
            Upsampled tensor, where out_channels is in_channels / 4,
            and spatial dimensions are doubled.
        """
        x = self.convgru(x, hidden_state)
        x = self.up(x)
        return x


class Decoder(nn.Module):
    """ConvGRU-based decoder that stacks multiple :class:`DecoderBlock` layers.

    After each block the spatial resolution is doubled via pixel-shuffle.

    Parameters
    ----------
    output_channels : int, optional
        Number of output channels. Default is ``1``.
    num_blocks : int, optional
        Number of decoder blocks to stack. Default is ``4``.
    **kwargs
        Additional keyword arguments forwarded to each :class:`DecoderBlock`.
    """

    def __init__(self, output_channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.channel_sizes = [output_channels * 4 ** (i + 1) for i in reversed(range(num_blocks))]
        self.blocks = nn.ModuleList(
            [DecoderBlock(self.channel_sizes[i], self.channel_sizes[i], **kwargs) for i in range(num_blocks)]
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch time hidden_channels height width"],
        hidden_states: list[Float[torch.Tensor, "batch _ _ _"]],
    ) -> Float[torch.Tensor, "batch time out_channels out_height out_width"]:
        """Forward the decoder through all blocks.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time hidden_channels height width"]
            Initial decoder input (usually zeros or noise).
        hidden_states : list of Float[torch.Tensor, "batch _ _ _"]
            Hidden states from the encoder (in reverse order), one per block.

        Returns
        -------
        out : Float[torch.Tensor, "batch time out_channels out_height out_width"]
            Output tensor at original spatial resolution.
        """
        for block, hidden_state in zip(self.blocks, hidden_states, strict=True):
            x = block(x, hidden_state)
        return x


class ConvGruModel(nn.Module):
    """Full encoder-decoder model for spatio-temporal forecasting.

    Encodes an input sequence into multi-scale hidden states and decodes
    them into a forecast sequence, optionally generating multiple ensemble
    members via noisy decoder inputs.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels. Default is ``1``.
    num_blocks : int, optional
        Number of encoder and decoder blocks. Default is ``4``.
    noisy_decoder : bool, optional
        If ``True``, feed random noise as decoder input. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to :class:`Encoder` and
        :class:`Decoder`.
    """

    def __init__(self, input_channels: int = 1, num_blocks: int = 4, noisy_decoder: bool = False, **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.noisy_decoder = noisy_decoder
        self.encoder = Encoder(input_channels, num_blocks, **kwargs)
        self.decoder = Decoder(input_channels, num_blocks, **kwargs)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch time channels height width"], steps: int, ensemble_size: int = 1
    ) -> Float[torch.Tensor, "batch steps _ height width"]:
        """Forward the encoder-decoder model.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch time channels height width"]
            Input sequence.
        steps : int
            Number of future timesteps to forecast.
        ensemble_size : int, optional
            Number of ensemble members to generate. When ``> 1``, the decoder
            is always run with noisy inputs. Default is ``1``.

        Returns
        -------
        preds : Float[torch.Tensor, "batch steps out_channels height width"]
            Forecast tensor.
        """
        _, _, _, H, W = x.shape
        divisor = 2**self.num_blocks
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        encoded = self.encoder(x)

        x_dec_shape = list(encoded[-1].shape)
        x_dec_shape[1] = steps

        last_hidden_per_block = [e[:, -1] for e in reversed(encoded)]

        if ensemble_size > 1:
            preds = []
            for _ in range(ensemble_size):
                x_dec = torch.randn(x_dec_shape, dtype=encoded[-1].dtype, device=encoded[-1].device)
                decoded = self.decoder(x_dec, last_hidden_per_block)
                preds.append(decoded)
            out = torch.cat(preds, dim=2)
        else:
            x_dec_func = torch.randn if self.noisy_decoder else torch.zeros
            x_dec = x_dec_func(x_dec_shape, dtype=encoded[-1].dtype, device=encoded[-1].device)
            out = self.decoder(x_dec, last_hidden_per_block)

        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]

        return out

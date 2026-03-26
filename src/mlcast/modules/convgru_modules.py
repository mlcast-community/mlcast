"""ConvGRU encoder-decoder architecture for spatio-temporal forecasting.

Provides the building blocks (ConvGRUCell, ConvGRU, Encoder, Decoder) and
the full EncoderDecoder model with optional ensemble generation via noisy
decoder inputs.
"""

import torch
import torch.nn as nn


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

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.combined_gates = conv_layer(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_layer(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, inpt: torch.Tensor | None = None, h_s: torch.Tensor | None = None) -> torch.Tensor:
        """Forward the ConvGRU cell for a single timestep.

        Parameters
        ----------
        inpt : torch.Tensor or None, optional
            Input tensor of shape ``(B, input_size, H, W)``.
        h_s : torch.Tensor or None, optional
            Hidden state tensor of shape ``(B, hidden_size, H, W)``.

        Returns
        -------
        new_state : torch.Tensor
            Updated hidden state of shape ``(B, hidden_size, H, W)``.
        """
        if h_s is None and inpt is None:
            raise ValueError("Both input and state can't be None")
        elif h_s is None:
            h_s = torch.zeros(
                inpt.size(0), self.hidden_size, inpt.size(2), inpt.size(3), dtype=inpt.dtype, device=inpt.device
            )
        elif inpt is None:
            inpt = torch.zeros(
                h_s.size(0), self.input_size, h_s.size(2), h_s.size(3), dtype=h_s.dtype, device=h_s.device
            )

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

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.cell = ConvGRUCell(input_size, hidden_size, kernel_size, conv_layer)

    def forward(self, x: torch.Tensor | None = None, h: torch.Tensor | None = None) -> torch.Tensor:
        """Unroll the ConvGRU cell over the sequence (time) dimension.

        Parameters
        ----------
        x : torch.Tensor or None, optional
            Input tensor of shape ``(B, T, input_size, H, W)``.
        h : torch.Tensor or None, optional
            Initial hidden state of shape ``(B, hidden_size, H, W)``.

        Returns
        -------
        hidden_states : torch.Tensor
            Stacked hidden states of shape ``(B, T, hidden_size, H, W)``.
        """
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

    def __init__(self, input_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.convgru = ConvGRU(input_size, input_size, kernel_size, conv_layer)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.

        Returns
        -------
        out : torch.Tensor
            Downsampled tensor of shape ``(B, T, C*4, H/2, W/2)``.
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward the encoder through all blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.

        Returns
        -------
        hidden_states : list of torch.Tensor
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

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.convgru = ConvGRU(input_size, hidden_size, kernel_size, conv_layer)
        self.up = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward the decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.
        hidden_state : torch.Tensor
            Hidden state from the corresponding encoder block.

        Returns
        -------
        out : torch.Tensor
            Upsampled tensor of shape ``(B, T, hidden_size // 4, H*2, W*2)``.
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

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Forward the decoder through all blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.
        hidden_states : list of torch.Tensor
            Hidden states from the encoder (in reverse order), one per block.

        Returns
        -------
        out : torch.Tensor
            Output tensor at original spatial resolution.
        """
        for block, hidden_state in zip(self.blocks, hidden_states, strict=True):
            x = block(x, hidden_state)
        return x


class EncoderDecoder(nn.Module):
    """Full encoder-decoder model for spatio-temporal forecasting.

    Encodes an input sequence into multi-scale hidden states and decodes
    them into a forecast sequence, optionally generating multiple ensemble
    members via noisy decoder inputs.

    Parameters
    ----------
    channels : int, optional
        Number of input/output channels. Default is ``1``.
    num_blocks : int, optional
        Number of encoder and decoder blocks. Default is ``4``.
    **kwargs
        Additional keyword arguments forwarded to :class:`Encoder` and
        :class:`Decoder`.
    """

    def __init__(self, channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.encoder = Encoder(channels, num_blocks, **kwargs)
        self.decoder = Decoder(channels, num_blocks, **kwargs)

    def forward(self, x: torch.Tensor, steps: int, noisy_decoder: bool = False, ensemble_size: int = 1) -> torch.Tensor:
        """Forward the encoder-decoder model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, C, H, W)``.
        steps : int
            Number of future timesteps to forecast.
        noisy_decoder : bool, optional
            If ``True``, feed random noise as decoder input. Default is ``False``.
        ensemble_size : int, optional
            Number of ensemble members to generate. When ``> 1``, the decoder
            is always run with noisy inputs. Default is ``1``.

        Returns
        -------
        preds : torch.Tensor
            Forecast tensor. Shape is ``(B, steps, C, H, W)`` when
            ``ensemble_size == 1``, or
            ``(B, steps, ensemble_size * C, H, W)`` when ``ensemble_size > 1``.
        """
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
            return torch.cat(preds, dim=2)
        else:
            x_dec_func = torch.randn if noisy_decoder else torch.zeros
            x_dec = x_dec_func(x_dec_shape, dtype=encoded[-1].dtype, device=encoded[-1].device)
            decoded = self.decoder(x_dec, last_hidden_per_block)
            return decoded

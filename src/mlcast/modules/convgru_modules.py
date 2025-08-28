import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        # update and reset gates are combined for optimization
        self.combined_gates = conv_layer(input_size + hidden_size, 2 * hidden_size, kernel_size, padding=padding)
        self.out_gate = conv_layer(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, inpt: torch.Tensor | None = None, h_s: torch.Tensor | None = None) -> torch.Tensor:
        """Forward the ConvGRU cell. If any of the input is None,
        it is initialized to zeros based on the shape of the other input.
        If both inputs are None, an error is raised.

        Args:
            inpt: input tensor (b, input_size, h, w)
            h_s: hidden state tensor (b, hidden_size, h, w)

            Returns:
                new hidden state tensor (b, hidden_size, h, w)

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


class ConvGRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.cell = ConvGRUCell(input_size, hidden_size, kernel_size, conv_layer)

    def forward(self, x: torch.Tensor | None = None, h: torch.Tensor | None = None) -> torch.Tensor:
        """Forward the ConvGRU cell over multiple elements in the sequence (timesteps).
        The input tensor x is expected to have the shape (b, seq_len, input_size, h, w)
        and the hidden state tensor h is expected to have the shape (b, hidden_size, h, w).

               x[:, 0]              x[:, 1]
                  |                    |
                  v                    v
               *------*             *------*
        h -->  | Cell | --> h_0 --> | Cell | --> h_1 ...
               *------*             *------*

        If any of the input is None,
        it is initialized to zeros based on the shape of the other input.
        If both inputs are None, an error is raised.

        Args:
            x: input tensor (b, seq_len, input_size, h, w)
            h: hidden state tensor (b, hidden_size, h, w)

            Returns:
                new hidden state tensor (b, seq_len, hidden_size, h, w)
                [h_0, h_1, h_2, ...]

        """
        h_s = []
        for i in range(x.size(1)):
            h = self.cell(x[:, i], h)
            h_s.append(h)
        return torch.stack(h_s, dim=1)


class EncoderBlock(nn.Module):
    """A ConvGRU-based Encoder block that stacks a ConvGRU layer and a reduction in the spatial
    dimensions by applying a pixel_unshuffle operation.
    """

    def __init__(self, input_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.convgru = ConvGRULayer(input_size, input_size, kernel_size, conv_layer)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the Encoder block. The input tensor x is expected to have the shape (b, seq_len, c, h, w).

        Args:
            x: input tensor (b, seq_len, c, h, w)

            Returns:
                new hidden state tensor (b, seq_len, hidden_size, h/2, w/2)

        """
        x = self.convgru(x)
        x = self.down(x)
        return x


class Encoder(nn.Module):
    r"""A ConvGRU-based Encoder that stacks multiple ConvGRU layers. After each ConvGRU layer, a reduction in the
    spatial dimensions is achieved by applying a pixel_unshuffle operation.

        ///    Ecnoder Block 1    \\\                   ///    Ecnoder Block 2    \\\
         /--------------------------------------------\ /--------------------------------------------\
        |                                              |                                              |
        *        *---------*      *-----------------*  *   *---------*      *-----------------*       *
    X -> | ConvGRU | ---> | Pixel Unshuffle | ---> | ConvGRU | ---> | Pixel Unshuffle | ---> ...
    |    *---------*  |   *-----------------*  |   *---------*  |   *-----------------*  |
    v                 v                        v                v                        v
          [x_0,x_1,...]     H0 [h0_0,h0_1,...]    H0 [h0_0,h0_1,...]    H1 [h1_0,h1_1,...]       H1 [h1_0,h1_1,...]
          (b, t, c, h, w)   (b, t, c, h, w)       (b, t, c*4, h/2, w/2) (b, t, c*4, h/2, w/2)    (b, t, c*16, h/4, w/4)

        Args:
    input_size (int): Number of input channels.
    num_blocks (int): Number of Encoder blocks.
    """

    def __init__(self, input_channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.channel_sizes = [input_channels * 4**i for i in range(num_blocks)]  # [1, 4, 16, 64]
        self.blocks = nn.ModuleList([EncoderBlock(self.channel_sizes[i], **kwargs) for i in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward the Encoder. The input tensor x is expected to have the shape (b, seq_len, c, h, w).

        Args:
            x: input tensor (b, seq_len, c, h, w)

            Returns:
                the list of hidden state tensors for all blocks [(b, seq_len, c*4, h/2, w/2),
                (b, seq_len, c*16, h/4, w/4), ...]

        """
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class DecoderBlock(nn.Module):
    """A ConvGRU-based Decoder block that stacks a ConvGRU layer and an expansion in the spatial
    dimensions by applying a pixel_shuffle operation.
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, conv_layer: nn.Module = nn.Conv2d):
        super().__init__()
        self.convgru = ConvGRULayer(input_size, hidden_size, kernel_size, conv_layer)
        self.up = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward the Decoder block. The input tensor x is expected to have the shape (b, seq_len, c, h, w).

        Args:
            x: input tensor (b, seq_len, c, h, w)
            hidden_state: hidden state tensor (b, hidden_size, h, w)

            Returns:
                new hidden state tensor (b, seq_len, hidden_size // 4, h*2, w*2)
        """
        x = self.convgru(x, hidden_state)
        x = self.up(x)
        return x


class Decoder(nn.Module):
    r"""A ConvGRU-based Decoder that stacks multiple ConvGRU layers. After each ConvGRU layer, an expansion in the
    spatial dimensions is achieved by applying a pixel_shuffle operation.
    All hidden sizes are computed based on the desired output features (number of channels in output in the last
    layer of decoder).

        Args:
    output_size (int): Number of output channels.
    num_blocks (int): Number of Decoder blocks.
    kwargs: Additional arguments for ConvGRU.
    """

    def __init__(self, output_channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.channel_sizes = [output_channels * 4 ** (i + 1) for i in reversed(range(num_blocks))]  # [256, 64, 16, 4]
        self.blocks = nn.ModuleList(
            [DecoderBlock(self.channel_sizes[i], self.channel_sizes[i], **kwargs) for i in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Forward the Decoder. The input tensor x is expected to have the shape (b, seq_len, c, h, w).

        Args:
            x: input tensor (b, seq_len, c, h, w)
            hidden_states: list of hidden state tensors for all blocks [(b, seq_len, c*4, h/2, w/2),
                (b, seq_len, c*16, h/4, w/4), ...]

            Returns:
                the output tensor (b, seq_len, output_channels, h*2^num_blocks, w*2^num_blocks)

        """
        for block, hidden_state in zip(self.blocks, hidden_states, strict=False):
            x = block(x, hidden_state)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, channels: int = 1, num_blocks: int = 4, **kwargs):
        super().__init__()
        self.encoder = Encoder(channels, num_blocks, **kwargs)
        self.decoder = Decoder(channels, num_blocks, **kwargs)

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        # encode the input tensor into a sequence of hidden states
        encoded = self.encoder(x)

        # create an empty tensor with the same shape as the last hidden state of the encoder to use as a dummy
        # input for the decoder
        x_shape = list(encoded[-1].shape)

        # set the desired number of timestep for the output
        x_shape[1] = steps
        x = torch.zeros(x_shape, dtype=encoded[-1].dtype, device=encoded[-1].device)

        # collect all the last hidden states of the encoder blocks in reverse order
        last_hidden_per_block = [e[:, -1] for e in reversed(encoded)]

        # decode the input tensor into a forecast sequence of N timesteps
        decoded = self.decoder(x, last_hidden_per_block)
        return decoded


ConvGRU = EncoderDecoder

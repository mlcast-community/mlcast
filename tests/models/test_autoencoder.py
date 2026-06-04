import torch
import torch.nn.functional as F

from mlcast.models.autoencoder import AutoencoderNet, Decoder, Encoder


def test_encoder_output_shape() -> None:
    """Encoder should preserve time and downsample spatial dimensions."""
    batch_size = 2
    input_steps = 4
    channels = 1
    height = 16
    width = 16
    latent_channels = 6

    encoder = Encoder(input_channels=channels, hidden_channels=4, latent_channels=latent_channels, num_blocks=2)
    x = torch.randn(batch_size, input_steps, channels, height, width)

    z = encoder(x)

    assert z.shape == (batch_size, latent_channels, input_steps, height // 4, width // 4)


def test_decoder_output_shape() -> None:
    """Decoder should preserve time and upsample spatial dimensions."""
    batch_size = 2
    input_steps = 4
    channels = 1
    latent_channels = 6
    latent_height = 4
    latent_width = 4

    decoder = Decoder(output_channels=channels, hidden_channels=4, latent_channels=latent_channels, num_blocks=2)
    z = torch.randn(batch_size, latent_channels, input_steps, latent_height, latent_width)

    y = decoder(z)

    assert y.shape == (batch_size, input_steps, channels, latent_height * 4, latent_width * 4)


def test_autoencoder_reconstruction_forward_pass() -> None:
    """Autoencoder should reconstruct tensors with the same shape as its input."""
    batch_size = 2
    input_steps = 3
    channels = 2
    height = 16
    width = 16

    encoder = Encoder(input_channels=channels, hidden_channels=4, latent_channels=8, num_blocks=2)
    decoder = Decoder(output_channels=channels, hidden_channels=4, latent_channels=8, num_blocks=2)
    model = AutoencoderNet(encoder=encoder, decoder=decoder)
    x = torch.randn(batch_size, input_steps, channels, height, width)

    y = model(x)

    assert y.shape == x.shape


def test_autoencoder_improves_reconstruction_loss() -> None:
    """Autoencoder should reduce reconstruction loss on a tiny generated dataset."""
    torch.manual_seed(42)
    batch_size = 8
    input_steps = 2
    channels = 1
    height = 8
    width = 8

    encoder = Encoder(input_channels=channels, hidden_channels=4, latent_channels=4, num_blocks=1)
    decoder = Decoder(output_channels=channels, hidden_channels=4, latent_channels=4, num_blocks=1)
    model = AutoencoderNet(encoder=encoder, decoder=decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    spatial_pattern = torch.linspace(-1.0, 1.0, height * width).reshape(1, 1, 1, height, width)
    temporal_scale = torch.linspace(0.5, 1.5, input_steps).reshape(1, input_steps, 1, 1, 1)
    samples = spatial_pattern * temporal_scale
    samples = samples.repeat(batch_size, 1, channels, 1, 1)

    with torch.no_grad():
        initial_loss = F.mse_loss(model(samples), samples).item()

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(samples), samples)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(model(samples), samples).item()

    assert final_loss < initial_loss

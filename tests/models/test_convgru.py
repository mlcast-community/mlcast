import torch

from mlcast.modules.convgru_modules import ConvGruModel


def test_convgru_dynamic_padding():
    """Verify that ConvGruModel dynamically pads non-power-of-2 inputs and crops the output."""
    # Given an input with awkward spatial dimensions
    batch_size = 2
    time_steps = 3
    channels = 1
    height = 250
    width = 250

    # 250 is not divisible by 2^4 (16).
    # The next multiple of 16 is 256. The model should pad to 256, run the forward pass,
    # and then crop the output back to 250x250.

    x = torch.randn(batch_size, time_steps, channels, height, width)

    model = ConvGruModel(input_channels=channels, num_blocks=4)
    model.eval()

    forecast_steps = 4

    with torch.no_grad():
        preds = model(x, steps=forecast_steps, ensemble_size=1)

    # Check that it didn't crash and the output shape is exactly (batch, steps, channels, height, width)
    # The single ensemble member case returns out_channels = channels.
    assert preds.shape == (batch_size, forecast_steps, channels, height, width)


def test_convgru_dynamic_padding_ensemble():
    """Verify that ConvGruModel dynamically pads non-power-of-2 inputs and crops the output for ensemble generation."""
    # Given an input with awkward spatial dimensions
    batch_size = 1
    time_steps = 2
    channels = 2
    height = 117
    width = 123

    x = torch.randn(batch_size, time_steps, channels, height, width)

    model = ConvGruModel(input_channels=channels, num_blocks=3)
    model.eval()

    forecast_steps = 2
    ensemble_size = 5

    with torch.no_grad():
        preds = model(x, steps=forecast_steps, ensemble_size=ensemble_size)

    # Check that it didn't crash and the output shape is exactly (batch, steps, ensemble_size * channels, height, width)
    # Actually wait: The decoder block outputs the same number of channels as the final upsampling step.
    # In the `ConvGruModel.forward` with `ensemble_size > 1`, `out` is `torch.cat(preds, dim=2)`.
    # Let's verify the exact channel dimension. The original output channels per ensemble member is `channels`.
    assert preds.shape == (batch_size, forecast_steps, channels * ensemble_size, height, width)

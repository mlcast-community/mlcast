import torch

from mlcast.models.convgru import ConvGruModel


def test_convgru_dynamic_padding() -> None:
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

    forecast_steps = 4
    model = ConvGruModel(input_steps=time_steps, forecast_steps=forecast_steps, input_channels=channels, num_blocks=4)
    model.eval()

    with torch.no_grad():
        preds = model(x)

    # Check that it didn't crash and the output shape is exactly (batch, steps, 1, channels, height, width)
    # The single ensemble member case adds an explicit ensemble dimension.
    assert preds.shape == (batch_size, forecast_steps, 1, channels, height, width)


def test_convgru_dynamic_padding_ensemble() -> None:
    """Verify that ConvGruModel dynamically pads non-power-of-2 inputs and crops the output for ensemble generation."""
    # Given an input with awkward spatial dimensions
    batch_size = 1
    time_steps = 2
    channels = 2
    height = 117
    width = 123

    x = torch.randn(batch_size, time_steps, channels, height, width)

    forecast_steps = 2
    ensemble_size = 5
    model = ConvGruModel(
        input_steps=time_steps,
        forecast_steps=forecast_steps,
        ensemble_size=ensemble_size,
        input_channels=channels,
        num_blocks=3,
    )
    model.eval()

    with torch.no_grad():
        preds = model(x)

    # Check that it didn't crash and the output shape has an explicit ensemble dimension:
    # (batch, forecast_steps, ensemble_size, channels, height, width)
    assert preds.shape == (batch_size, forecast_steps, ensemble_size, channels, height, width)


def test_convgru_rejects_wrong_input_steps() -> None:
    """ConvGruModel should reject inputs that violate its configured input length."""
    model = ConvGruModel(input_steps=3, forecast_steps=2, input_channels=1, num_blocks=2)
    x = torch.randn(1, 2, 1, 32, 32)

    try:
        model(x)
    except ValueError as exc:
        assert "Expected 3 input timesteps" in str(exc)
    else:
        raise AssertionError("Expected ConvGruModel to reject wrong input_steps")

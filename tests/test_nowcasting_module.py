import numpy as np
import torch

from mlcast.modules.forecasting import OutputSpaceForecastingTaskModule


class DummyForecastNetwork(torch.nn.Module):
    """Minimal fixed-shape forecasting network for module tests."""

    def __init__(self, input_steps: int, forecast_steps: int, ensemble_size: int = 1) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.ensemble_size = ensemble_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels, height, width = x.shape
        return torch.zeros(
            batch_size, self.forecast_steps, self.ensemble_size, channels, height, width, device=x.device
        )


def test_nowcasting_module_forward_uses_network_shape_contract() -> None:
    """OutputSpaceForecastingTaskModule should call fixed-shape forecasting networks as network(x)."""
    network = DummyForecastNetwork(input_steps=3, forecast_steps=5, ensemble_size=2)
    module = OutputSpaceForecastingTaskModule(network=network, loss_class="crps")
    x = torch.randn(4, 3, 1, 8, 8)

    preds = module(x)

    assert preds.shape == (4, 5, 2, 1, 8, 8)


def test_nowcasting_module_predict_uses_configured_output_shape() -> None:
    """Prediction horizon and ensemble size should come from the configured network."""
    network = DummyForecastNetwork(input_steps=3, forecast_steps=4, ensemble_size=2)
    module = OutputSpaceForecastingTaskModule(network=network, loss_class="crps")
    past = torch.ones(3, 8, 8)

    preds = module.predict(past, standard_name="rainfall_rate")

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, 4, 1, 8, 8)

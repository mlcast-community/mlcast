import numpy as np
import torch

from mlcast.models.autoencoder import AutoencoderNet, Decoder, Encoder
from mlcast.models.diffusion import ConditionerNet, DenoiserUNet, DiffusionScheduler, LatentDiffusionNet
from mlcast.modules.forecasting import LatentDiffusionTaskModule, OutputSpaceForecastingTaskModule
from mlcast.modules.reconstruction import ReconstructionTaskModule


class IdentityReconstructionNetwork(torch.nn.Module):
    """Minimal reconstruction network used in wrapper tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""
        return x


def test_reconstruction_module_uses_batch_as_target() -> None:
    """ReconstructionTaskModule should compute loss against the input batch itself."""
    module = ReconstructionTaskModule(network=IdentityReconstructionNetwork(), loss_class="mse")
    batch = torch.randn(2, 3, 1, 4, 4)

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_output_space_forecasting_task_module_trainable_parameters_match_network() -> None:
    """OutputSpaceForecastingTaskModule should optimize the forecasting network parameters."""

    class TinyForecastNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    network = TinyForecastNetwork()
    module = OutputSpaceForecastingTaskModule(network=network, loss_class="mse")

    assert module.trainable_parameters == list(network.parameters())


def _build_autoencoder() -> AutoencoderNet:
    encoder = Encoder(input_channels=1, hidden_channels=4, latent_channels=4, num_blocks=1)
    decoder = Decoder(output_channels=1, hidden_channels=4, latent_channels=4, num_blocks=1)
    return AutoencoderNet(encoder=encoder, decoder=decoder)


def _build_diffusion_net() -> LatentDiffusionNet:
    conditioner = ConditionerNet(latent_channels=4, hidden_channels=8, num_blocks=1)
    denoiser = DenoiserUNet(latent_channels=4, condition_channels=8, hidden_channels=8, num_blocks=2)
    scheduler = DiffusionScheduler(timesteps=2)
    return LatentDiffusionNet(conditioner=conditioner, denoiser=denoiser, scheduler=scheduler)


def test_latent_diffusion_module_training_step_runs() -> None:
    """LatentDiffusionTaskModule should encode forecasting batches and return scalar loss."""
    module = LatentDiffusionTaskModule(
        autoencoder=_build_autoencoder(), diffusion_net=_build_diffusion_net(), forecast_steps=3
    )
    batch = {
        "input": torch.randn(2, 2, 1, 8, 8),
        "target": torch.randn(2, 3, 1, 8, 8),
    }

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_latent_diffusion_task_module_trainable_parameters_exclude_autoencoder() -> None:
    """LatentDiffusionTaskModule should optimize only diffusion-net parameters."""
    autoencoder = _build_autoencoder()
    diffusion_net = _build_diffusion_net()
    module = LatentDiffusionTaskModule(autoencoder=autoencoder, diffusion_net=diffusion_net, forecast_steps=3)

    assert module.trainable_parameters == list(diffusion_net.parameters())
    assert module.trainable_parameters != list(autoencoder.parameters())


def test_latent_diffusion_module_predict_uses_configured_output_shape() -> None:
    """LatentDiffusionTaskModule prediction should decode configured ensemble forecasts."""
    module = LatentDiffusionTaskModule(
        autoencoder=_build_autoencoder(),
        diffusion_net=_build_diffusion_net(),
        forecast_steps=3,
        ensemble_size=2,
    )
    past = torch.ones(2, 8, 8)

    preds = module.predict(past, standard_name="rainfall_rate")

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, 3, 1, 8, 8)

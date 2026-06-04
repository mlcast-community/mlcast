import torch

from mlcast.models.diffusion import (
    ConditionerNet,
    DenoiserUNet,
    DiffusionLoss,
    DiffusionScheduler,
    LatentDiffusionNet,
)


def _build_diffusion_net(latent_channels: int = 1, hidden_channels: int = 8, timesteps: int = 4) -> LatentDiffusionNet:
    conditioner = ConditionerNet(latent_channels=latent_channels, hidden_channels=hidden_channels, num_blocks=1)
    denoiser = DenoiserUNet(
        latent_channels=latent_channels,
        condition_channels=hidden_channels,
        hidden_channels=hidden_channels,
        num_blocks=1,
    )
    scheduler = DiffusionScheduler(timesteps=timesteps)
    return LatentDiffusionNet(conditioner=conditioner, denoiser=denoiser, scheduler=scheduler)


def test_latent_diffusion_net_api() -> None:
    """LatentDiffusionNet should predict noise with the target latent shape."""
    input_time = 2
    forecast_steps = 3
    latent_channels = 1
    height = 4
    width = 4
    diffusion_net = _build_diffusion_net(latent_channels=latent_channels, hidden_channels=4, timesteps=2)
    noised_target = torch.randn(2, latent_channels, forecast_steps, height, width)
    input_latents = torch.randn(2, latent_channels, input_time, height, width)
    timesteps = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        predicted_noise = diffusion_net(noised_target, timesteps, input_latents)

    assert predicted_noise.shape == noised_target.shape


def test_diffusion_model_improves_loss_on_generated_latents() -> None:
    """Diffusion model should reduce noise-prediction loss on generated latents."""
    torch.manual_seed(7)
    batch_size = 8
    latent_channels = 1
    input_time = 2
    forecast_time = 3
    height = 4
    width = 4
    diffusion_net = _build_diffusion_net(latent_channels=latent_channels, hidden_channels=8, timesteps=1)
    loss_fn = DiffusionLoss(diffusion_net)
    optimizer = torch.optim.Adam(diffusion_net.parameters(), lr=5e-3)

    input_latents = torch.randn(batch_size, latent_channels, input_time, height, width)
    target_base = input_latents.mean(dim=2, keepdim=True)
    target_latents = target_base.repeat(1, 1, forecast_time, 1, 1)
    target_latents = target_latents + 0.05 * torch.randn_like(target_latents)

    torch.manual_seed(42)
    with torch.no_grad():
        initial_loss = loss_fn(input_latents, target_latents).item()

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(input_latents, target_latents)
        loss.backward()
        optimizer.step()

    torch.manual_seed(42)
    with torch.no_grad():
        final_loss = loss_fn(input_latents, target_latents).item()

    assert final_loss < initial_loss

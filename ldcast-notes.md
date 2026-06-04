# LDCast implementation notes

Architecture decisions and differences between reference implementations.

## EMA scope

| Reference | Scope | Decay |
|-----------|-------|-------|
| **DMI** | Full `LatentDiffusionNet` (conditioner + denoiser + scheduler buffers) | `0.9999` |
| **Martinbo** | Denoiser submodule only (`diffusion_net.denoiser`) | `0.9999` |
| **Ours** | Full `LatentDiffusionNet` (matches DMI) | `0.9999` |

**Rationale for full-network EMA**: The conditioner is a single-pass feed-forward network
called once per sample, so weight noise matters less than in the denoiser. However, there
is no downside to smoothing it too, and it keeps the code simpler (EMA wraps the entire
diffusion net rather than reaching into a private submodule). Full-network EMA is the
standard practice in DDPM, Stable Diffusion, and DMI's reference.

If denoiser-only EMA were desired in the future, the change is in
`forecasting.py:514`:

```python
# Full network (current, matches DMI):
self.ema = ExponentialMovingAverage(diffusion_net, decay=ema_decay)

# Denoiser only (Martinbo):
self.ema = ExponentialMovingAverage(diffusion_net.denoiser, decay=ema_decay)
```

## Optimizer

| Reference | Type | Betas | Weight decay | Autoencoder LR | Diffusion LR |
|-----------|------|-------|-------------|----------------|--------------|
| **DMI** | `AdamW` | `(0.5, 0.9)` | `1e-3` | `1e-3` | `1e-4` |
| **Martinbo** | `AdamW` | `[0.5, 0.9]` | `0.001` | `1e-3` | `1e-4` |
| **Ours** | `AdamW` | `(0.5, 0.9)` | `1e-3` | `1e-3` | `1e-4` |

Both references agree on all optimizer settings.

## LR scheduler

`ReduceLROnPlateau(factor=0.25, patience=3)` for both stages. Monitor metric naming
differs: DMI uses `val_rec_loss` / `val_loss_ema`, Martinbo uses `val/rec_loss` /
`val/loss` (TensorBoard convention). Ours follows Martinbo's naming.

## Diffusion noise schedule

Both references use `timesteps=1000` with `beta_start=1e-4, beta_end=2e-2`.
`DiffusionScheduler` defaults already match these beta bounds.

## Monitor metric naming

DMI uses underscores (`val_rec_loss`), Martinbo uses TensorBoard-style slashes
(`val/rec_loss`). We follow Martinbo / TensorBoard convention — slashes give
automatic grouping in the TensorBoard UI.

## Early stopping

DMI and Martinbo both use `patience=6`. Martinbo adds `check_finite=False` on the
diffusion stage. We follow both.

## Batch size

None of the three implementations agree on batch size:
- DMI: `batch_size=4` (autoencoder) / `1` (diffusion) — example configs
- Martinbo: `batch_size=1` for both stages
- Ours: `batch_size=4` (autoencoder) / `1` (diffusion) — matches DMI

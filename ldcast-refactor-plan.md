# Latent Diffusion Refactor Plan

0. Config naming and CLI contract
- [x] Rename `training_experiment` to `convgru_training_experiment`.
- [x] Do not keep `training_experiment` as an alias.
- [x] Reserve `latent_diffusion_experiment` as the top-level config name for the two-stage workflow.
- [x] Require `mlcast train` users to provide an explicit base config via `--config config:<name>` or `--config /path/to/config.yaml`.
- [x] Update CLI help text to list the included config entry points explicitly.
- [x] Update all docs, examples, tests, and scripts to use `convgru_training_experiment` instead of `training_experiment`.
- [x] Treat `convgru_training_experiment` as the existing ConvGRU forecasting example, not as a special default config.

1. Forecasting and reconstruction data
- [x] Move the existing sampled-sequence source-data logic into `src/mlcast/data/sequence.py`.
- [x] Rename `SourceDataDatasetBase`, `SourceDataPrecomputedSamplingDataset`, and `SourceDataRandomSamplingDataset` to `SourceDataSequenceDatasetBase`, `SourceDataPrecomputedSequenceDataset`, and `SourceDataRandomSequenceDataset` under the sequence data area.
- [x] Remove the old source-data public API rather than keeping compatibility re-exports.
- [x] Keep the existing sampled-sequence implementation as the source-data sequence layer.
- [x] Sequence datasets should own normalization and return normalized tensors of shape `(sequence_steps, channels, height, width)`.
- [x] Replace forecasting-specific sampling parameters in the source-data sequence layer with a single `sequence_steps` parameter.
- [x] Add `src/mlcast/data/forecasting.py`.
- [x] Add a generic `ForecastingDataset` that wraps a base sequence dataset, takes `input_steps` and `forecast_steps`, validates `input_steps + forecast_steps == sequence_steps`, and returns forecasting samples.
- [x] `ForecastingDataset` should derive `target_mask` itself rather than relying on the base sequence dataset to return masks.
- [x] Add `src/mlcast/data/reconstruction.py`.
- [x] Add `ReconstructionDataset`, a generic wrapper around a base sequence dataset that slices each full sequence into all overlapping windows of length `input_steps` and returns only the tensor window.
- [x] Add `src/mlcast/data/datamodules.py`.
- [x] Rename `SourceDataDataModule` to `ForecastingDataModule` in `src/mlcast/data/datamodules.py`.
- [x] `ForecastingDataModule` should remain factory-based and build `ForecastingDataset` instances over the underlying sequence datasets.
- [x] Add `ReconstructionDataModule` to `src/mlcast/data/datamodules.py`; it remains factory-based, builds the underlying sequence datasets, splits them into train/val/test, and wraps each split with `ReconstructionDataset`.
- [x] Keep this generic: no LDCast-specific naming in the module or class names.
- [x] Forecasting should stay one-sequence-to-one-sample.
- [x] Reconstruction should expand each sequence into `sequence_steps - input_steps + 1` overlapping samples.
- [x] Stage 1 should use reconstruction windows of length `input_steps` derived from the full sequence dataset.

2. Autoencoder model architecture
- Autoencoder model split:
  - [x] `src/mlcast/models/autoencoder/encoder.py` for `Encoder` and `EncoderBlock`.
  - [x] `src/mlcast/models/autoencoder/decoder.py` for `Decoder` and `DecoderBlock`.
  - [x] `src/mlcast/models/autoencoder/net.py` for `AutoencoderNet`.
- [x] Use `input_steps` for the stage-1 reconstruction window length; do not introduce names like `autoenc_time_ratio`.
- Autoencoder validation and tests:
  - [x] encoder output shape.
  - [x] decoder output shape.
  - [x] autoencoder reconstruction forward pass.
  - [x] autoencoder improves reconstruction loss on a small generated dataset after a few training steps.

3. Forecasting model contract
- [x] Standardize all forecasting models on init-time `input_steps`, `forecast_steps`, and `ensemble_size`.
- [x] Standardize forecasting model inference on `forward(x)` only; do not pass `forecast_steps` or `ensemble_size` at runtime.
- [x] Refactor the existing ConvGRU path to follow this fixed-shape contract.
- [x] Add config consistency checks that dataset `input_steps` and `forecast_steps` match the configured forecasting model.

4. Diffusion model architecture
- Diffusion model split:
  - [x] `src/mlcast/models/diffusion/conditioner.py` for latent conditioning blocks and `ConditionerNet`.
  - [x] `src/mlcast/models/diffusion/denoiser.py` for `DenoiserUNet` and timestep-aware helpers.
  - [x] `src/mlcast/models/diffusion/net.py` for `LatentDiffusionNet`.
  - [x] `src/mlcast/models/diffusion/scheduler.py`, `ema.py`, `sampler.py`, `loss.py` for diffusion support code.
- Validation and tests:
  - [x] latent diffusion model API.
  - [x] diffusion model improves loss on a small generated latent dataset after a few training steps.

5. Task modules (Lightning modules)
- [x] Add `src/mlcast/modules/forecasting.py`, introduce `BaseForecastingTaskModule`, and rename `NowcastLightningModule` to `OutputSpaceForecastingTaskModule`.
- [x] `BaseForecastingTaskModule` should own optimizer/scheduler plumbing, while each concrete task module defines which parameters are trainable.
- [x] `OutputSpaceForecastingTaskModule` should optimize the forecasting network parameters.
- [x] Remove runtime `forecast_steps` and `ensemble_size` arguments from the forecasting task module and its `predict()` API.
- [x] Add `src/mlcast/modules/reconstruction.py` with a generic `ReconstructionTaskModule` for any reconstruction model.
- [x] Add a `LatentDiffusionTaskModule` that owns the trained autoencoder, optimizes only the diffusion-network parameters, trains diffusion in latent space, and handles decoded forecast inference.
- [x] Keep `modules/` for task-level Lightning modules only; keep `models/` for pure architectures.

6. Training experiment
- [x] Add a new two-stage `LatentDiffusionTrainingExperiment` (initially called `LDCastTrainingExperiment`).
- [x] Keep `convgru_training_experiment` as the existing ConvGRU forecasting example and one of the explicitly selected included CLI configs.
- [x] Stage 1 builds the reconstruction dataset, autoencoder model, and `ReconstructionTaskModule`, then trains the autoencoder.
- [x] Stage 2 reuses the same trained in-memory autoencoder instance, builds the diffusion dataset/model/`LatentDiffusionTaskModule`, then trains latent diffusion.
- [x] Stage 2 freezes the reused autoencoder parameters and optimizes only the latent diffusion task module's diffusion-network parameters.
- [x] The shared Fiddle graph should define the autoencoder once and reference the same object in both stages, but no unresolved Fiddle objects should flow into actual `torch.nn.Module.__init__` calls.
- [x] Stage-2 diffusion training uses the trained encoder to produce input and target latents; the trained decoder is retained for final forecast decoding but is not used in the stage-2 diffusion loss.
- [x] Reuse the same forecasting dataset abstraction in stage 2; do not add a separate latent dataset layer.
- [x] Add tests for shared object identity and stage sequencing.

7. Audit and migration targets
- [x] Update CLI/help text in `src/mlcast/__main__.py` to require an explicit base config and list the included config entry points.
- [x] Rename `training_experiment` to `convgru_training_experiment` in `src/mlcast/config/base.py` and export it from `src/mlcast/config/__init__.py`.
- [x] Add the LDCast config entry point to `src/mlcast/config/__init__.py` alongside the existing ConvGRU example config.
- [x] Keep `src/mlcast/config/orchestrator.py` compatible with both the existing single-stage `Experiment` and `LatentDiffusionTrainingExperiment` through a common `run()` surface.

- [x] Keep existing ConvGRU CLI/config tests passing while adding separate tests for the two-stage config explicitly.

- [ ] Add real but small-scale end-to-end tests with generated sample data for the autoencoder stage, diffusion stage, and full two-stage sequencing.
- [x] Align metric naming with TensorBoard conventions: use `/` as hierarchy separator and `rec_loss`/`loss` to distinguish stages.

## 8. Align latent diffusion config with DMI/Martinbo reference

The `ldcast-dmi/` reference implementation differs from our current
`latent_diffusion_experiment` config in several ways. Changes below would
align us more closely with DMI.

### Optimizer
- **DMI**: `AdamW` with `lr=1e-3` (autoenc) / `1e-4` (diffusion),
  `betas=(0.5, 0.9)`, `weight_decay=1e-3` for **both** stages.
- **Ours**: `Adam` with `lr=1e-4` for both stages, default betas, no
  weight decay.
- **To align**: switch to `AdamW`, use DMI betas/weight_decay, and raise
  autoencoder LR to `1e-3`.

### LR scheduler
- **DMI**: `ReduceLROnPlateau(factor=0.25, patience=3)`, monitors
  `val_rec_loss` (autoenc) / `val_loss_ema` (diffusion).
- **Ours**: `ReduceLROnPlateau(factor=0.5, patience=10)`, monitors
  `val_loss` for both stages.
- **To align**: reduce factor to `0.25` and patience to `3`; use separate
  monitor metrics per stage (autoenc → `val_loss`, diffusion → `val_loss`).

### Learning rate warmup
- **DMI**: Linear warmup support in diffusion stage (`lr_warmup`, default
  0 — disabled). Autoencoder has none.
- **Ours**: No warmup in either stage.
- **To align**: no change needed unless LR warmup is desired.

### EMA
- **DMI**: `LitEma` with `decay=0.9999` (adaptive based on num_updates),
  only on diffusion model weights. EMA weights swapped in for
  validation/testing.
- **Ours**: `ExponentialMovingAverage` with `decay=0.999` for diffusion
  net, swapped in for val/test.
- **To align**: increase EMA decay to `0.9999`.

### Early stopping
- **DMI**: patience `6`, monitors `val_rec_loss` / `val_loss_ema`,
  `check_finite=False` on diffusion.
- **Ours**: patience `20`, monitors `val_loss`.
- **To align**: reduce patience to `6`; consider `check_finite=False`.

### Model checkpointing
- **DMI**: `save_top_k=3`, monitors `val_rec_loss` / `val_loss_ema`.
- **Ours**: `save_top_k=1`, monitors `val_loss`.
- **To align**: increase save_top_k to `3`.

### Diffusion noise schedule
- **DMI**: `timesteps=1000`, linear beta schedule from `1e-4` to `2e-2`.
- **Ours**: `timesteps=20`, default linear schedule.
- **To align**: increase to `timesteps=1000` and match beta range.

### Batch size and gradient accumulation
- **DMI**: `batch_size=4` (autoenc, example) / `batch_size=1` (diffusion,
  example); `accumulate_grad_batches=2`.
- **Ours**: `batch_size=16` / `8`; no gradient accumulation.
- **To align**: reduce batch sizes and add `accumulate_grad_batches=2`.

### DDP strategy
- **DMI**: `DDPStrategy(find_unused_parameters=True)` on autoencoder.
- **Ours**: default (no `DDPStrategy`).
- **To align**: no change needed unless running DDP.

## Martinbo alignment notes

The `feat/ldcast-martinbo` branch differs from both our current config and
the DMI reference in several ways.

### Optimizer
- **DMI**: `AdamW`, `lr=1e-3` / `1e-4`, `betas=(0.5, 0.9)`, `wd=1e-3`.
- **Martinbo**: `AdamW`, `lr=1e-3` / `1e-4`, `betas=[0.5, 0.9]`, `wd=0.001`.
- **Ours**: `Adam`, `lr=1e-4` for both, default betas, no weight decay.
- **To align**: Martinbo matches DMI exactly — `AdamW`, per-stage LR, betas, and wd.

### LR scheduler
- **DMI**: `ReduceLROnPlateau(factor=0.25, patience=3)`, monitors
  `val_rec_loss` / `val_loss_ema`.
- **Martinbo**: `ReduceLROnPlateau(factor=0.25, patience=3)`, monitors
  `val/rec_loss` / `val/loss`.
- **Ours**: `ReduceLROnPlateau(factor=0.5, patience=10)`, monitors
  `val_loss` for both stages.
- **To align**: Martinbo matches DMI's factor/patience; only monitor-metric
  naming differs (`val/rec_loss` vs `val_rec_loss`).

### Learning rate warmup
- **DMI**: Diffusion warmup support (`lr_warmup=0`, disabled by default).
- **Martinbo**: No warmup support in either stage.
- **Ours**: No warmup in either stage.
- **To align**: no change needed (DMI also has it disabled by default).

### EMA
- **DMI**: `LitEma` with `decay=0.9999` (adaptive), on full diffusion model.
- **Martinbo**: `EMA` with `decay=0.9999` (dynamic, adaptive), wraps
  **denoiser only** (`store_device='cuda'`).
- **Ours**: `ExponentialMovingAverage` with `decay=0.999`, on diffusion net.
- **To align**: increase decay to `0.9999`; consider whether EMA should wrap
  the full diffusion net or just the denoiser.

### Early stopping
- **DMI**: patience `6`, monitors `val_rec_loss` / `val_loss_ema`,
  `check_finite=False` on diffusion.
- **Martinbo**: patience `6`, monitors `val/loss_epoch` (both stages),
  `check_finite=False`.
- **Ours**: patience `20`, monitors `val_loss`.
- **To align**: Martinbo matches DMI's patience and `check_finite=False`;
  monitor naming differs (`val/loss_epoch` vs `val_loss_ema`).

### Model checkpointing
- **DMI**: `save_top_k=3`, monitors `val_rec_loss` / `val_loss_ema`.
- **Martinbo**: Not explicitly configured in branch `config.yaml` (relies on
  Lightning default, `save_top_k=1`).
- **Ours**: `save_top_k=1`, monitors `val_loss`.
- **To align**: Martinbo implicitly matches Ours on `save_top_k`; DMI differs
  with `save_top_k=3`.

### Diffusion noise schedule
- **DMI**: `timesteps=1000`, linear beta `1e-4` to `2e-2`.
- **Martinbo**: `timesteps=1000`, linear beta `1e-4` to `2e-2` (defaults,
  config section is `{}`).
- **Ours**: `timesteps=20`, default linear schedule.
- **To align**: Martinbo matches DMI exactly — `timesteps=1000`, same beta range.

### Batch size and gradient accumulation
- **DMI**: `batch_size=4` / `1` (example configs), `accumulate_grad_batches=2`.
- **Martinbo**: `batch_size=1` for both stages; no `accumulate_grad_batches`.
- **Ours**: `batch_size=16` / `8`; no gradient accumulation.
- **To align**: Martinbo uses smaller batches than both DMI and Ours; none
  of the three agree on batch size strategy.

### DDP strategy
- **DMI**: `DDPStrategy(find_unused_parameters=True)` (autoenc) /
  `DDPStrategy()` (diffusion).
- **Martinbo**: `strategy='ddp'` (string), `sync_batchnorm=True`, `num_nodes=1`.
- **Ours**: default (no `DDPStrategy`).
- **To align**: no change needed unless running DDP.

### Diffusion parameterization and loss
- **DMI**: `parameterization="eps"`, `loss_type="l2"` (MSE).
- **Martinbo**: `parametrization="eps"` (note: spelling difference),
  `nn.MSELoss()`.
- **Ours**: `parameterization="eps"` in `DiffusionLoss` (L2 via
  `nn.MSELoss` reduction).
- **To align**: All three agree on `eps` + MSE — no change needed.


## 9. Possible future work

Architecture and feature upgrades not covered by section 8 (config alignment).
Each entry explains why it might be worth doing.

### VAE autoencoder (KL-regularised latent space)

DMI uses `AutoencoderKL` — a VAE trained with a KL-divergence loss on the latent
distribution, producing a smoother, more Gaussian latent space. Ours uses a
deterministic autoencoder.

**Why it might be a good idea**: Diffusion models assume the target distribution
is Gaussian (they start from Gaussian noise and reverse-diffuse). A KL-
regularised latent space is closer to Gaussian, which can make diffusion easier
and improve sample quality. It also enables latent-space interpolation and
manipulation. However, it adds training complexity (KL loss weighting, posterior
collapse risk).

### Larger denoiser (3D UNet with cross-attention)

DMI's `UNetModel` uses 128 model channels, attention blocks at multiple
resolutions, 8 attention heads, and 3D convolutions over (time, height, width).
Ours uses 32 hidden channels, no attention, and preserves time as a plain
channel dimension.

**Why it might be a good idea**: More capacity → better fit to complex
precipitation patterns. Cross-attention allows the denoiser to selectively
attend to conditioning context at each resolution, which is more expressive
than our simple input concatenation. The cost is larger memory footprint and
longer training times.

### Multi-resolution context encoder (AFNO cascade)

DMI's `AFNONowcastNetCascade` produces a feature pyramid where each spatial
resolution has its own channel depth. The denoiser selects the appropriate level
based on its current spatial resolution. Ours uses a single-resolution
`ConditionerNet` that is interpolated if spatial sizes don't match.

**Why it might be a good idea**: Multi-resolution conditioning lets the denoiser
access fine-grained local information at high resolutions and broad context at
low resolutions simultaneously. This is standard in modern conditional diffusion
models (e.g. Stable Diffusion's cross-attention to CLIP embeddings at multiple
scales).

### PLMS accelerated sampling

DMI's `PLMSSampler` uses Adams-Bashforth multistep integration to reduce 1000
DDPM steps to ~50 with minimal quality loss. Ours uses a basic ancestral DDPM
sampler that iterates over all timesteps.

**Why it might be a good idea**: 20× faster inference is critical for
operational nowcasting where latency matters. PLMS is well-established (from
CompVis latent-diffusion) and requires no retraining — it works with any
trained eps-predicting model.

### Adaptive EMA decay (LitEma-style)

DMI uses `LitEma` with adaptive decay `min(0.9999, (1+n)/(10+n))` where `n` is
the number of EMA updates. Ours uses a fixed decay of `0.9999`.

**Why it might be a good idea**: Adaptive decay starts lower (giving more weight
to recent parameters early in training when they change fastest) and converges
to 0.9999. This accelerates early training while maintaining the benefits of EMA
at convergence. Simple to implement — just change the decay formula in `update`.

### Multiple beta schedules (cosine, sqrt)

DMI supports linear, cosine, sqrt_linear, and sqrt schedules. Ours supports
linear only.

**Why it might be a good idea**: Cosine schedules (from Nichol & Dhariwal,
"Improved DDPM") add noise more gradually, which can improve sample quality —
especially at low resolutions or with fewer timesteps. Having multiple schedules
also enables hyperparameter search. The `DiffusionScheduler` would need to
accept a `schedule` string and dispatch to the right formula.

### x0 parameterization and L1 loss

DMI supports predicting `x0` (the clean target) instead of `eps` (the noise),
and using L1 instead of L2 for the loss. Ours uses `eps` + L2 only.

**Why it might be a good idea**: `x0` prediction can be more stable at high
noise levels and is required for certain sampling techniques. L1 loss tends to
produce sharper outputs (less blurring than L2), which is desirable for
precipitation fields with sharp rain/no-rain boundaries.

### Classifier-free guidance (CFG) in the sampler

DMI's `PLMSSampler` supports CFG via an `unconditional_guidance_scale`
parameter. Ours has no CFG support.

**Why it might be a good idea**: CFG lets the user trade off ensemble diversity
vs. forecast fidelity at inference time by scaling the conditional prediction
away from the unconditional prediction. Higher guidance → more confident
(less diverse) forecasts. This is useful operational flexibility.

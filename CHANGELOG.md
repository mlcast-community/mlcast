# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Dataset sampling folded into `mlcast` from the standalone `mlcast-dataset-sampler` as
  `mlcast.data.source_data.sampling`. `mlcast build-sampling-index` scans a source Zarr (on CPU, or GPU with
  `--device cuda`) and writes a parquet sampling index of candidate datacubes with per-datacube stats
  (`nan_count`, `sum`, `mean`, `frac_wet`) and the sampling parameters in its metadata;
  `mlcast validate-sampling-index` checks a file against that contract.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- Pluggable candidate selectors (`CandidateSelector`, `ImportanceSelector`, `UniformSelector`, with a
  `register_selector`/`get_selector` registry) that pick the subset of the index to train on, once at dataset
  init. `SourceDataDataModule` takes per-split `train_selector` and `eval_selector` arguments; the default
  experiment uses importance sampling on train and uniform 10% sampling on val/test.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- CI check that fails pull requests which don't update `CHANGELOG.md`; PRs that don't need an entry can opt out
  with the `no changelog` label. [#26](https://github.com/mlcast-community/mlcast/pull/26) @leifdenby

### Changed

- **Breaking:** `SourceDataPrecomputedSamplingDataset` is renamed `SourceDataIndexedDataset` and its
  `csv_path` argument is renamed `index_path`. It now reads a parquet sampling index; legacy `.csv` indexes are
  still accepted and used as-is.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- **Breaking:** the default experiment now loads `rainfall_flux` (was `rainfall_rate`) from
  `./data/sampled_datacubes.parquet`.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- **Breaking:** `target_mask` is now collapsed over the whole sequence to shape `(1, C, H, W)`. A cell is valid
  only if it is finite at every input and target step, and the masked loss broadcasts the mask over the forecast
  steps. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- `bottleneck`, `pyarrow` and `pydantic` are now core dependencies; the unused `fire` dependency is removed.
  [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- The CLI and `mlcast.data` import lazily, so `mlcast -h` and the data-prep commands start in well under a
  second. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn

### Fixed

- Time-split subsets of an indexed dataset now rebase the absolute `t` indexes onto the sliced store and drop
  windows that would cross the split boundary, so they no longer read the wrong time steps or leak samples
  across splits. [#17](https://github.com/mlcast-community/mlcast/pull/17) @franchg @leifdenby @noralinn
- Missing changelog entries for [#17](https://github.com/mlcast-community/mlcast/pull/17).
  [#26](https://github.com/mlcast-community/mlcast/pull/26) @leifdenby
- Register `equivalent_reflectivity_factor` (radar reflectivity in dBZ) in the normalization registries, so
  training on it no longer fails with a `KeyError`, and drop the non-CF `radar_reflectivity` name from the
  sampler. [#24](https://github.com/mlcast-community/mlcast/pull/24) @leifdenby

## v0.1.0 - 2026-06-01

This release establishes `mlcast` as a usable foundation for machine-learning weather nowcasting experiments.
It brings together configurable experiment construction, Zarr-backed meteorological data loading, a PyTorch
Lightning training loop, probabilistic ConvGRU nowcasting, and reproducible run configuration. With this
release, users can train the built-in ConvGRU baseline from the command line, customize experiments through
Fiddle configs and fiddlers, or replace the network with their own nowcasting architecture while keeping the
surrounding data, training, logging, and reproducibility machinery.

### Added

- Fiddle-based experiment configuration centered on the default `training_experiment` graph, covering dataset
  setup, data module, network, Lightning module, optimizer, scheduler, callbacks, logger, and trainer.
- `mlcast train` CLI for launching the default experiment, applying ordered `set:` overrides and `fiddler:`
  mutations, loading saved YAML configs, switching config functions, and inspecting resolved configs before
  training.
- Python API for constructing, editing, validating, building, and running experiment configs programmatically.
- Zarr-backed source-data pipeline with CF-standard-name variable lookup, normalization, precomputed CSV
  sampling, random on-the-fly sampling, train/validation/test splitting, spatial augmentations, optional target
  masks, and anonymous S3 access.
- Precipitation conversion and normalization utilities for rainfall rate, rainfall flux, and 5-minute rainfall
  amount.
- Generic PyTorch Lightning nowcasting module that wraps any compatible network and handles training,
  validation, testing, optimizer/scheduler construction, ensemble forecasting, masked losses, image logging,
  checkpoint loading, and prediction.
- Built-in ConvGRU encoder-decoder nowcasting model with multi-scale spatial encoding/decoding, latent-space
  forecast rollout, padding for arbitrary spatial sizes, and optional stochastic ensemble generation.
- Loss functions for deterministic and probabilistic nowcasting, including MSE, MAE, CRPS, almost-fair CRPS,
  temporal consistency regularization, and masking of invalid target pixels.
- Semantic fiddlers for common multi-parameter configuration changes, including random sampling, variable
  selection, masking, ratio splits, anonymous S3 datasets, and MLflow logging.
- Reproducibility and experiment tracking support through saved or uploaded Fiddle YAML configs, flattened config
  hyperparameters, TensorBoard logging, MLflow logging, W&B-compatible config artifact handling, and MLflow
  system metadata/metrics logging.
- Documentation covering installation, CLI workflows, Python workflows, custom network integration, available
  fiddlers, project structure, and the ConvGRU architecture.

### Changed

- Training now centers on a single Fiddle-based experiment graph rather than the earlier ad hoc configuration
  flow.
- The CLI now uses `mlcast train` with nested `--config` overrides and named fiddlers for more reproducible
  experiment changes.
- The default training setup now uses the ConvGRU ensemble nowcasting pipeline as the primary supported
  baseline.
- Saved experiment configs are now persisted alongside runs for exact reproduction.

## 0.0.1a4 - 2025-08-28

This was the first published PyPI release of `mlcast`, establishing the initial installable package baseline with
the early nowcasting training pipeline, core package layout, and initial public documentation.

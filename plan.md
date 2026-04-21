# Implementation Plan: Fiddle Orchestrator Pattern & Generalized Architecture

## Phase 0: Testing Setup (TDD Prerequisites)
- [x] **Dependencies**: Add `pytest`, `pytest-cov`, `s3fs`, `fsspec`, `aiohttp`, and `cf_xarray` to `pyproject.toml` (in `optional-dependencies.dev` or main `dependencies` as appropriate).
- [x] **Test Fixtures**: Create `tests/conftest.py` with a session-scoped fixture `italian_dataset` that explicitly downloads and caches the Italian DPC dataset from S3 locally, restricted to the first 100 timesteps using `.isel(time=slice(0, 100))` to ensure offline compatibility and avoid `simplecache` corruption bugs.

## Phase 1: Package Restructuring & Cleanup (PR Feedback)
- [x] **Tests**: Create `tests/data/test_normalization.py` and `tests/test_losses.py`.
   - [x] Verify `rainrate_to_normalized` and its inverse.
   - [x] Verify `AFCRPS` and `CRPS` loss shapes.
- [x] **Rename & Move**: Move `src/mlcast/utils.py` to `src/mlcast/data/normalization.py`. Update all imports.
- [x] **Normalization Registry**: Create a `NORMALIZATION_REGISTRY` dict in `normalization.py` mapping CF standard names (e.g., `"rainfall_rate"`) to their respective normalization functions.
- [x] **Visualization**: Extract `apply_radar_colormap` and `log_images` out of the model file into a new `src/mlcast/visualization.py`.
- [x] **Losses (`src/mlcast/losses.py`)**:
   - [x] Rename `afCRPS` to `AFCRPS`.
   - [x] Update `build_loss` signature to explicitly default to `loss_class="mse"`.
   - [x] Expand docstrings to include explicit expected tensor shapes.
- [x] **CLI (`__main__.py`)**:
   - [x] Refactor `__main__.py` to use `argparse` subparsers instead of manual argv checking. Make `train` automatically default to `training_experiment` if no base config is provided.

- [x] **Config YAML (`src/mlcast/configs.py`)**:
   - [x] Rename `convgru_experiment` to `training_experiment`.
   - [x] Change the default `TensorBoardLogger` name from `"convgru"` to `"mlcast"`.
   - [x] Retain `config_to_dict` since Fiddle lacks a native recursive YAML serializer.

## Phase 2A: Decoupling the Architecture (Structural)
- [x] **Generic Lightning Wrapper (`src/mlcast/models/base.py` or similar)**:
   - [x] Refactor `RadarLightningModel` to be a general-purpose wrapper that takes an injected PyTorch `nn.Module` (the network architecture) instead of hardcoding `EncoderDecoder`.
- [x] **Network Architectures**:
   - [x] Ensure the raw `EncoderDecoder` logic acts cleanly as an interchangeable module (e.g., rename to `ConvGruModel` or expose it cleanly).

## Phase 2B: Typing and Documentation
- [x] Add rigorous NumPy-style docstrings to all methods in `base.py` and `convgru_modules.py`.
- [x] Add `jaxtyping` annotations to all `torch.nn.Module` classes to strictly define input/output tensor shapes and dtypes.

## Phase 3: Decoupling the Data Layer (Generalized Source Data)
- [ ] **Tests**: Create `tests/data/test_source_datasets.py` and `tests/data/test_data_module.py`.
   - [ ] Test datasets using the `italian_dataset` fixture to ensure they output `(Time, Channels, Height, Width)` and use `cf_xarray` correctly.
   - [ ] Test DataModule splits with a mock dataset factory.
- [ ] **Rename Files & Classes**:
   - [ ] Rename `src/mlcast/data/zarr_datamodule.py` to `source_data_module.py` and `RadarDataModule` to `SourceDataDataModule`.
   - [ ] Rename `src/mlcast/data/zarr_dataset.py` to `source_datasets.py`.
   - [ ] Rename `SampledRadarDataset` to `SourceDataPrecomputedSamplingDataset`.
- [ ] **New Dataset**:
   - [ ] Create `SourceDataRandomSamplingDataset` which performs on-the-fly random spatial and temporal slicing.
- [ ] **Dataset Implementation Details**:
   - [ ] Update both datasets' `__init__` to accept `standard_names: list[str]`.
   - [ ] Rename internal xarray reference from `self.zg` to standard `self.ds`.
   - [ ] Update `__getitem__` to use `self.ds.cf[std_name]` to extract arrays, and `NORMALIZATION_REGISTRY[std_name]` to normalize them, before stacking along the channel dimension.
- [ ] **DataModule Implementation Details**:
   - [ ] Inject `dataset_factory` (a `Callable[..., Dataset]`) into `SourceDataDataModule.__init__` so it doesn't hardcode dataset instantiation.

## Phase 4: Enhancing Model Robustness
- [ ] **Tests**: Create `tests/models/test_convgru.py`.
   - [ ] Pass dummy tensors of "awkward" non-power-of-2 sizes (e.g., `250x250`) and assert the dynamic padding handles it gracefully without crashing.
- [ ] **Architecture (`ConvGruModel`)**:
   - [ ] Move the dynamic padding logic currently in `predict()` into the `forward()` pass using `torch.nn.functional.pad`.
   - [ ] Ensure it mathematically pads to `2 ** num_blocks` before the encoder, and crops it back before returning.

## Phase 5: The Fiddle Config & Validation (The Orchestrator)
- [ ] **Tests**: Create `tests/test_configs.py`.
   - [ ] Generate a `base_experiment` config, break contracts intentionally, and assert `train_from_config` raises `ValueError`.
- [ ] **`src/mlcast/configs.py` (Base Config)**:
   - [ ] Define the `dataset_factory` using `fdl.Partial`.
   - [ ] Inject both the `dataset_factory` into the `DataModule` and the network into the generic `LightningModule`.
- [ ] **`src/mlcast/configs.py` (Validation in `train_from_config`)**:
   - [ ] Contract 1: Network `input_channels` == `len(dataset_factory.standard_names)`.
   - [ ] Contract 2: Dataset `width` must be divisible by `2 ** network.num_blocks`.
   - [ ] Contract 3: Dataset `steps` > `forecast_steps`.
   - [ ] Contract 4: Ensemble models require CRPS or AFCRPS.

## Phase 6: Fiddlers for UX & Synchronization
- [ ] **Tests**: Update `tests/test_configs.py`.
   - [ ] Test that Fiddlers (`set_variables`, `toggle_masking`) successfully mutate the dataset and the model simultaneously.
- [ ] **`src/mlcast/configs.py` (Fiddlers)**:
   - [ ] Create `set_variables(cfg, standard_names: list[str])` to sync the dataset and the network's `input_channels`.
   - [ ] Create `toggle_masking(cfg, enabled: bool)` to sync `cfg.data.dataset_factory.return_mask` with `cfg.pl_module.masked_loss`.

## Phase 7: Documentation
- [ ] **Docs (`README.md`)**:
   - [ ] Add "Design" section to `README.md` with Mermaid class diagram (Separation of Concerns, Generalized Data Layer, Fiddle Orchestrator).
   - [ ] Add "Usage" section to `README.md` detailing default training, simple Fiddle overrides, semantic mutators (Fiddlers), and explicit examples for changing source dataset, variables, and dataset class. Highlight the serialized `config.yaml` saved in `logs/mlcast/`.

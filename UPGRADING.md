# Upgrading

This guide is a best effort at listing the steps needed to upgrade between `mlcast` releases. For the full list
of changes, see [CHANGELOG.md](CHANGELOG.md).

## v0.1.0 → v0.2.0

- Install with a PyTorch extra: `uv sync --extra cpu` (or `gpu-cu126`, `gpu-cu128`, `gpu-cu130`), or
  `pip install "mlcast[cpu]"`. PyTorch is no longer a core dependency, so a plain `uv sync` or
  `pip install mlcast` leaves it out.
- Rename `SourceDataPrecomputedSamplingDataset` to `SourceDataIndexedDataset` and its `csv_path` argument to
  `index_path`. Regenerate sampling indexes as parquet with `mlcast build-sampling-index` (legacy `.csv` indexes
  are still read as-is).
- The default experiment now loads `rainfall_flux` (was `rainfall_rate`) from `./data/sampled_datacubes.parquet`.
- `target_mask` now has shape `(1, C, H, W)`, collapsed over the whole sequence.

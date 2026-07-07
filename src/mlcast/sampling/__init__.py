"""Datacube sampling support for mlcast.

This subpackage holds the contract and helpers shared between the offline
*dataset sampler* (which scans a source Zarr and writes a per-datacube stats
parquet) and the training-time dataset that consumes that parquet:

- :mod:`mlcast.sampling.stats_spec` — the canonical stats-parquet contract
  (column schema + validated metadata), read by the precomputed-sampling
  dataset instead of re-parsing a filename.
- :mod:`mlcast.sampling.samplers` — pluggable candidate-selection schemes
  (``Sampler`` + ``SAMPLER_REGISTRY``), e.g. :class:`ImportanceSampler`.
- :mod:`mlcast.sampling.units` — rain-rate vs reflectivity classification and
  default wet-pixel thresholds, from CF attributes.
"""

from .samplers import (
    SAMPLER_REGISTRY,
    ImportanceSampler,
    Sampler,
    UniformSampler,
    get_sampler,
    register_sampler,
)
from .stats_spec import (
    STATS_SCHEMA,
    StatsMetadata,
    ValidationReport,
    read_metadata,
    validate_stats_parquet,
)
from .units import default_wet_threshold, detect_data_kind

__all__ = [
    "SAMPLER_REGISTRY",
    "STATS_SCHEMA",
    "ImportanceSampler",
    "Sampler",
    "StatsMetadata",
    "UniformSampler",
    "ValidationReport",
    "default_wet_threshold",
    "detect_data_kind",
    "get_sampler",
    "read_metadata",
    "register_sampler",
    "validate_stats_parquet",
]

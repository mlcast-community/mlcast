"""Sampling of source datasets: building, checking and selecting from a sampling index.

A *sampling index* is a parquet file listing every candidate datacube of a
source Zarr (its corner ``t, x, y``) together with its *sample stats*
(``nan_count``, ``sum``, ``mean``, ``frac_wet``); the *sampling parameters*
used to build it are stored in the file's metadata. It is written offline by
``mlcast build-sampling-index`` and consumed at training time by
:class:`~mlcast.data.source_data_datasets.SourceDataIndexedDataset`.

- :mod:`.sampling_index_spec` — the canonical sampling-index contract (column
  schema + validated sampling parameters), read by the dataset instead of
  re-parsing a filename.
- :mod:`.selection` — pluggable candidate selectors (``CandidateSelector`` +
  ``SELECTOR_REGISTRY``), e.g. :class:`ImportanceSelector`.
- :mod:`.units` — rain-rate vs reflectivity classification and default
  wet-pixel thresholds, from CF attributes.
- :mod:`.commands` — the ``build-sampling-index`` and
  ``validate-sampling-index`` CLI commands.
"""

from .sampling_index_spec import (
    SAMPLING_INDEX_SCHEMA,
    SamplingParameters,
    ValidationReport,
    read_sampling_parameters,
    validate_sampling_index,
)
from .selection import (
    SELECTOR_REGISTRY,
    CandidateSelector,
    ImportanceSelector,
    UniformSelector,
    get_selector,
    register_selector,
)
from .units import default_wet_threshold, detect_data_kind

__all__ = [
    "SELECTOR_REGISTRY",
    "SAMPLING_INDEX_SCHEMA",
    "ImportanceSelector",
    "CandidateSelector",
    "SamplingParameters",
    "UniformSelector",
    "ValidationReport",
    "default_wet_threshold",
    "detect_data_kind",
    "get_selector",
    "read_sampling_parameters",
    "register_selector",
    "validate_sampling_index",
]

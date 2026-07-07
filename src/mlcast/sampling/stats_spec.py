"""Canonical contract for a stats parquet file.

A "stats parquet" is the output of the dataset sampler's ``stats`` command:
one row per surviving datacube candidate, plus schema-level JSON metadata
carrying the sampling parameters. Both halves of that contract live here:

- ``STATS_SCHEMA`` — the column layout, which the sampler writes from.
- ``StatsMetadata`` — a pydantic model of the sampling parameters, with
  field- and cross-field validation. Consumers read it via
  :func:`read_metadata` instead of re-parsing a filename.

:func:`validate_stats_parquet` checks an arbitrary file against both.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

SCHEMA_VERSION = 1
STATS_METADATA_KEY = b"mlcast.stats"

#: Single source of truth for the column layout of a stats parquet.
STATS_SCHEMA: pa.Schema = pa.schema(
    [
        ("t", pa.int32()),
        ("x", pa.int32()),
        ("y", pa.int32()),
        ("nan_count", pa.int32()),
        ("sum", pa.float32()),
        ("mean", pa.float32()),
        ("frac_wet", pa.float32()),
    ]
)

#: Column names in canonical order, derived from STATS_SCHEMA.
STAT_COLUMNS: tuple[str, ...] = tuple(STATS_SCHEMA.names)


class StatsMetadata(BaseModel):
    """Sampling parameters carried in a stats parquet's schema metadata.

    Field names match the JSON keys stored under ``mlcast.stats``. Build from
    a raw dict with ``StatsMetadata.model_validate(payload)`` (unknown keys are
    ignored); serialise with ``model_dump(mode="json")``. Constraints are
    declared on the field types; the only imperative rules are cross-field.
    """

    model_config = ConfigDict(frozen=True)

    zarr_path: str
    data_var: str
    time_var: str
    start_date: datetime
    end_date: datetime
    time_step_minutes: PositiveInt
    time_depth: PositiveInt
    width: PositiveInt
    height: PositiveInt
    step_t: PositiveInt
    step_x: PositiveInt
    step_y: PositiveInt
    max_nan: NonNegativeInt
    wet_threshold: NonNegativeFloat
    data_kind: Literal["rainrate", "reflectivity"]  # mirrors units.DEFAULT_WET_THRESHOLD
    units: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def total_px(self) -> int:
        """Number of pixels in one datacube (time_depth * width * height)."""
        return self.time_depth * self.width * self.height

    @model_validator(mode="after")
    def _check_cross_field(self) -> StatsMetadata:
        if self.max_nan > self.total_px:
            raise ValueError(f"max_nan ({self.max_nan}) exceeds datacube size ({self.total_px})")
        if self.start_date > self.end_date:
            raise ValueError(f"start_date ({self.start_date}) is after end_date ({self.end_date})")
        return self


def build_schema(metadata: StatsMetadata) -> pa.Schema:
    """Return STATS_SCHEMA with this file's metadata attached.

    This is what the sampler hands to its ``ParquetWriter`` so the column
    layout and the metadata payload come from one place.
    """
    payload = metadata.model_dump(mode="json")
    encoded = {STATS_METADATA_KEY: json.dumps(payload, sort_keys=True).encode()}
    return STATS_SCHEMA.with_metadata(encoded)


def read_metadata(path: str) -> StatsMetadata:
    """Load and validate the ``mlcast.stats`` metadata from a parquet file.

    Raises
    ------
    KeyError
        If the file carries no ``mlcast.stats`` metadata key.
    pydantic.ValidationError
        If the payload is malformed or violates a constraint.
    """
    schema = pq.read_schema(path)
    if schema.metadata is None or STATS_METADATA_KEY not in schema.metadata:
        raise KeyError(
            f"{path}: no 'mlcast.stats' metadata found in parquet schema. "
            f"Not a stats file produced by the mlcast dataset sampler?"
        )
    payload = json.loads(schema.metadata[STATS_METADATA_KEY].decode())
    return StatsMetadata.model_validate(payload)


class ValidationReport(BaseModel):
    """Result of validating a stats parquet file."""

    path: str
    errors: list[str] = []
    warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_stats_parquet(path: str, *, check_data: bool = True) -> ValidationReport:
    """Validate a stats parquet file against the canonical contract.

    Always checks the column schema (names + Arrow dtypes) and the metadata
    payload. When ``check_data`` is true, additionally reads the columns and
    asserts per-row value invariants. Returns a :class:`ValidationReport`;
    never raises for contract violations (only for unreadable files).
    """
    report = ValidationReport(path=path)

    # --- Metadata (ValidationError is a ValueError subclass) ------------------
    meta: StatsMetadata | None = None
    try:
        meta = read_metadata(path)
    except KeyError as e:
        report.errors.append(str(e))
    except ValueError as e:
        report.errors.append(f"metadata: {e}")

    if meta is not None and meta.schema_version != SCHEMA_VERSION:
        report.warnings.append(f"schema_version {meta.schema_version} != current {SCHEMA_VERSION}")

    # --- Column schema --------------------------------------------------------
    schema = pq.read_schema(path)
    actual = dict(zip(schema.names, schema.types, strict=True))
    missing = [c for c in STATS_SCHEMA.names if c not in actual]
    if missing:
        report.errors.append(f"missing columns: {missing}")
    for name in STATS_SCHEMA.names:
        expected = STATS_SCHEMA.field(name).type
        if name in actual and actual[name] != expected:
            report.errors.append(f"column {name!r} has dtype {actual[name]}, expected {expected}")

    if not check_data or missing:
        return report

    # --- Per-row value sanity (full read) ------------------------------------
    table = pq.read_table(path, columns=list(STATS_SCHEMA.names))
    if table.num_rows == 0:
        report.warnings.append("file has zero rows")
        return report
    cols = {name: table.column(name).to_numpy() for name in STATS_SCHEMA.names}

    for name in ("t", "x", "y", "nan_count"):
        if (cols[name] < 0).any():
            report.errors.append(f"column {name!r} has negative values")

    nan_count = cols["nan_count"]
    if meta is not None:
        if (nan_count > meta.total_px).any():
            report.errors.append(f"nan_count exceeds datacube size {meta.total_px} for some rows")
        if (nan_count > meta.max_nan).any():
            report.errors.append(
                f"nan_count exceeds metadata max_nan ({meta.max_nan}) for some rows "
                f"(the stats command applies this as a hard filter)"
            )

    frac_wet = cols["frac_wet"]
    if (~((frac_wet >= 0.0) & (frac_wet <= 1.0))).any():
        report.errors.append("frac_wet has values outside [0, 1]")
    for name in ("sum", "mean", "frac_wet"):
        if np.isinf(cols[name]).any():
            report.errors.append(f"column {name!r} contains +/-inf")

    return report

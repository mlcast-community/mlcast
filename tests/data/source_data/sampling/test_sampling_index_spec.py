"""Tests for the sampling-index contract: schema, metadata, validation."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from mlcast.data.source_data.sampling.sampling_index_spec import (
    SAMPLING_INDEX_SCHEMA,
    SAMPLING_PARAMETERS_KEY,
    SCHEMA_VERSION,
    SamplingParameters,
    build_schema,
    read_sampling_parameters,
    validate_sampling_index,
)


def _good_params(**overrides):
    params = {
        "zarr_path": "/data/it-dpc.zarr",
        "data_var": "RR",
        "time_var": "time",
        "start_date": "2020-01-01T00:00:00",
        "end_date": "2020-12-31T00:00:00",
        "time_step_minutes": 5,
        "time_depth": 24,
        "width": 256,
        "height": 256,
        "step_t": 3,
        "step_x": 16,
        "step_y": 16,
        "max_nan": 10000,
        "wet_threshold": 0.1,
        "data_kind": "rainrate",
        "units": "mm/h",
    }
    params.update(overrides)
    return params


def _write_parquet(path, metadata: SamplingParameters, rows: dict):
    schema = build_schema(metadata)
    batch = pa.record_batch([pa.array(rows[c]) for c in SAMPLING_INDEX_SCHEMA.names], schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd") as w:
        w.write_batch(batch)


def _good_rows(meta: SamplingParameters, n: int = 5):
    total = meta.total_px
    # All nan_count <= max_nan (the hard filter) and < total_px, so every
    # window has valid pixels and a finite mean — what build-sampling-index emits.
    nan_count = np.array([0, 1, 100, meta.max_nan // 2, meta.max_nan], dtype=np.int32)[:n]
    valid = total - nan_count
    mean = np.where(valid > 0, 0.5, np.nan).astype(np.float32)
    return {
        "t": (np.arange(n, dtype=np.int32) * meta.step_t),
        "x": (np.arange(n, dtype=np.int32) * meta.step_x),
        "y": (np.arange(n, dtype=np.int32) * meta.step_y),
        "nan_count": nan_count,
        "sum": (mean * valid).astype(np.float32),
        "mean": mean,
        "frac_wet": np.full(n, 0.3, dtype=np.float32),
    }


# --- SamplingParameters model ------------------------------------------------------


def test_metadata_roundtrip():
    meta = SamplingParameters.model_validate(_good_params())
    assert meta.total_px == 24 * 256 * 256
    assert meta.schema_version == SCHEMA_VERSION
    again = SamplingParameters.model_validate(meta.model_dump(mode="json"))
    assert again == meta


def test_metadata_ignores_unknown_keys():
    meta = SamplingParameters.model_validate(_good_params(some_future_field=123))
    assert not hasattr(meta, "some_future_field")


@pytest.mark.parametrize(
    "override",
    [
        {"time_depth": 0},
        {"width": -1},
        {"max_nan": -5},
        {"data_kind": "snowfall"},
        {"wet_threshold": -0.1},
        {"start_date": "2021-01-01T00:00:00", "end_date": "2020-01-01T00:00:00"},
        {"max_nan": 24 * 256 * 256 + 1},
    ],
)
def test_metadata_rejects_bad_values(override):
    with pytest.raises(ValidationError):
        SamplingParameters.model_validate(_good_params(**override))


def test_metadata_missing_required_key():
    params = _good_params()
    del params["data_kind"]
    with pytest.raises(ValidationError):
        SamplingParameters.model_validate(params)


# --- File validation ----------------------------------------------------------


def test_validate_clean_file(tmp_path):
    meta = SamplingParameters.model_validate(_good_params())
    path = str(tmp_path / "stats.parquet")
    _write_parquet(path, meta, _good_rows(meta))

    report = validate_sampling_index(path)
    assert report.ok, report.errors
    assert read_sampling_parameters(path) == meta


def test_validate_detects_bad_frac_wet(tmp_path):
    meta = SamplingParameters.model_validate(_good_params())
    rows = _good_rows(meta)
    rows["frac_wet"] = rows["frac_wet"].copy()
    rows["frac_wet"][0] = 1.5
    path = str(tmp_path / "stats.parquet")
    _write_parquet(path, meta, rows)

    report = validate_sampling_index(path)
    assert not report.ok
    assert any("frac_wet" in e for e in report.errors)


def test_validate_detects_nan_count_over_max(tmp_path):
    meta = SamplingParameters.model_validate(_good_params())
    rows = _good_rows(meta)
    rows["nan_count"] = rows["nan_count"].copy()
    rows["nan_count"][0] = meta.max_nan + 1
    path = str(tmp_path / "stats.parquet")
    _write_parquet(path, meta, rows)

    report = validate_sampling_index(path)
    assert not report.ok
    assert any("max_nan" in e for e in report.errors)


def test_validate_detects_wrong_dtype(tmp_path):
    meta = SamplingParameters.model_validate(_good_params())
    # Build a schema where 't' is int64 instead of int32.
    bad_schema = pa.schema(
        [("t", pa.int64())] + [(n, SAMPLING_INDEX_SCHEMA.field(n).type) for n in SAMPLING_INDEX_SCHEMA.names[1:]]
    ).with_metadata(build_schema(meta).metadata)
    rows = _good_rows(meta)
    path = str(tmp_path / "stats.parquet")
    batch = pa.record_batch(
        [pa.array(rows[c], type=bad_schema.field(c).type) for c in SAMPLING_INDEX_SCHEMA.names],
        schema=bad_schema,
    )
    with pq.ParquetWriter(path, bad_schema) as w:
        w.write_batch(batch)

    report = validate_sampling_index(path)
    assert not report.ok
    assert any("dtype" in e and "t'" in e for e in report.errors)


def test_validate_missing_metadata_key(tmp_path):
    # A parquet with correct columns but no mlcast.stats metadata.
    meta = SamplingParameters.model_validate(_good_params())
    rows = _good_rows(meta)
    path = str(tmp_path / "plain.parquet")
    batch = pa.record_batch([pa.array(rows[c]) for c in SAMPLING_INDEX_SCHEMA.names], schema=SAMPLING_INDEX_SCHEMA)
    with pq.ParquetWriter(path, SAMPLING_INDEX_SCHEMA) as w:
        w.write_batch(batch)

    report = validate_sampling_index(path)
    assert not report.ok
    assert any("mlcast.stats" in e for e in report.errors)


def test_validate_corrupt_metadata_payload(tmp_path):
    meta = SamplingParameters.model_validate(_good_params())
    rows = _good_rows(meta)
    bad_meta = {SAMPLING_PARAMETERS_KEY: json.dumps({"time_depth": -1}).encode()}
    schema = SAMPLING_INDEX_SCHEMA.with_metadata(bad_meta)
    path = str(tmp_path / "corrupt.parquet")
    batch = pa.record_batch([pa.array(rows[c]) for c in SAMPLING_INDEX_SCHEMA.names], schema=schema)
    with pq.ParquetWriter(path, schema) as w:
        w.write_batch(batch)

    report = validate_sampling_index(path)
    assert not report.ok
    assert any("metadata" in e for e in report.errors)

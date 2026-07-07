from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import xarray as xr

from mlcast.data.source_data_datasets import (
    SourceDataIndexedDataset,
    SourceDataRandomSamplingDataset,
)
from mlcast.sampling import ImportanceSampler, UniformSampler
from mlcast.sampling.stats_spec import StatsMetadata, build_schema


@pytest.fixture
def mock_csv(tmp_path: Path) -> str:
    """Create a temporary CSV file with coordinates."""
    df = pd.DataFrame(
        {
            "t": [0, 5, 10],
            "x": [10, 20, 30],
            "y": [10, 20, 30],
        }
    )
    csv_path = tmp_path / "coords.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_parquet(tmp_path: Path) -> str:
    """Create a temporary stats parquet (the sampler's output) with a mean column."""
    meta = StatsMetadata(
        zarr_path="dummy.zarr",
        data_var="RR",
        time_var="time",
        start_date="2016-01-01",
        end_date="2016-12-31",
        time_step_minutes=5,
        time_depth=2,
        width=16,
        height=16,
        step_t=1,
        step_x=1,
        step_y=1,
        max_nan=0,
        wet_threshold=0.1,
        data_kind="rainrate",
        units="mm/h",
    )
    table = pa.table(
        {
            "t": pa.array([0, 5, 10], pa.int32()),
            "x": pa.array([10, 20, 30], pa.int32()),
            "y": pa.array([10, 20, 30], pa.int32()),
            "nan_count": pa.array([0, 0, 0], pa.int32()),
            "sum": pa.array([1.0, 2.0, 3.0], pa.float32()),
            "mean": pa.array([0.05, 2.0, 10.0], pa.float32()),
            "frac_wet": pa.array([0.0, 0.1, 0.5], pa.float32()),
        },
        schema=build_schema(meta),
    )
    parquet_path = tmp_path / "stats.parquet"
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_indexed_sampling_dataset_parquet(fp_test_dataset: Path, mock_parquet: str) -> None:
    """Reads a stats parquet index; with no sampler the full pool is used."""
    ds = SourceDataIndexedDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_parquet,
        standard_names=["rainfall_flux"],
        input_steps=2,
        forecast_steps=1,
        width=16,
        height=16,
        return_mask=True,
    )

    assert len(ds) == 3  # all candidates, as-is
    sample = ds[0]
    assert sample["input"].shape == (2, 1, 16, 16)
    assert sample["target"].shape == (1, 1, 16, 16)


def test_indexed_importance_selection_is_fixed_and_keeps_extremes(fp_test_dataset: Path, mock_parquet: str) -> None:
    """ImportanceSampler selects a fixed, reproducible subset that keeps extremes."""
    kwargs = dict(
        zarr_path=str(fp_test_dataset),
        index_path=mock_parquet,
        standard_names=["rainfall_flux"],
        input_steps=2,
        forecast_steps=1,
        width=16,
        height=16,
        sampler=ImportanceSampler(),
        sampling_seed=0,
    )
    ds = SourceDataIndexedDataset(**kwargs)

    # a subset of the 3 candidates, with the wettest (t=10) always kept
    assert 1 <= len(ds) <= 3
    assert 10 in ds.coords["t"].to_numpy()
    # reproducible: same seed -> identical kept set
    ds2 = SourceDataIndexedDataset(**kwargs)
    assert np.array_equal(ds.coords["t"].to_numpy(), ds2.coords["t"].to_numpy())


def test_indexed_importance_sampler_requires_mean_column(fp_test_dataset: Path, mock_csv: str) -> None:
    """ImportanceSampler rejects a CSV index that has no mean column."""
    with pytest.raises(ValueError, match="mean"):
        SourceDataIndexedDataset(
            zarr_path=str(fp_test_dataset),
            index_path=mock_csv,
            standard_names=["rainfall_flux"],
            input_steps=2,
            forecast_steps=1,
            width=16,
            height=16,
            sampler=ImportanceSampler(),
        )


def test_indexed_uniform_sampler_works_on_csv(fp_test_dataset: Path, mock_csv: str) -> None:
    """A non-importance sampler (uniform) needs no mean column, so works on a CSV."""
    ds = SourceDataIndexedDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        input_steps=2,
        forecast_steps=1,
        width=16,
        height=16,
        sampler=UniformSampler(keep_fraction=1.0),
    )
    assert len(ds) == 3  # keep_fraction=1.0 -> the whole (3-row) index


def test_indexed_sampling_dataset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Test that SourceDataIndexedDataset outputs the correct shape."""
    input_steps = 2
    forecast_steps = 1
    ds = SourceDataIndexedDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        width=16,
        height=16,
        return_mask=True,
    )

    assert len(ds) == 3
    sample = ds[0]

    assert "input" in sample
    assert "target" in sample
    assert "target_mask" in sample

    input_t = sample["input"]
    target_t = sample["target"]
    target_mask_t = sample["target_mask"]

    assert input_t.shape == (input_steps, 1, 16, 16)
    assert target_t.shape == (forecast_steps, 1, 16, 16)
    assert target_mask_t.shape == (1, 1, 16, 16)  # collapsed over the sequence
    assert isinstance(input_t, torch.Tensor)
    assert isinstance(target_t, torch.Tensor)
    assert isinstance(target_mask_t, torch.Tensor)


def test_indexed_sampling_dataset_time_subset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Subset keeps only index rows whose full window fits the subset, and
    rebases their absolute ``t`` onto the sliced (0-based) store."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    # mock_csv has t = [0, 5, 10]; subset [3, 21) with time_depth 3 drops t=0
    # (before the start) and keeps t=5, 10, rebased onto the slice to [2, 7].
    ds = SourceDataIndexedDataset(
        zarr_path=str(fp_test_dataset),
        index_path=mock_csv,
        standard_names=["rainfall_flux"],
        input_steps=2,
        forecast_steps=1,
        time_depth=3,
        subset={"time": (str(time_index[3]), str(time_index[20]))},
    )
    assert len(ds) == 2
    assert sorted(ds.coords["t"].tolist()) == [2, 7]


def test_indexed_sampling_dataset_forecast_steps_guard(fp_test_dataset: Path, mock_csv: str) -> None:
    """Test that instantiation with input_steps=0 raises ValueError."""
    with pytest.raises(ValueError, match="input_steps"):
        SourceDataIndexedDataset(
            zarr_path=str(fp_test_dataset),
            index_path=mock_csv,
            standard_names=["rainfall_flux"],
            input_steps=0,
            forecast_steps=3,
        )


def test_random_sampling_dataset(fp_test_dataset: Path) -> None:
    """Test that SourceDataRandomSamplingDataset outputs the correct shape."""
    input_steps = 3
    forecast_steps = 2
    ds = SourceDataRandomSamplingDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        width=32,
        height=32,
        epoch_size=10,
        return_mask=True,
    )

    assert len(ds) == 10
    sample = ds[0]

    assert "input" in sample
    assert "target" in sample
    assert "target_mask" in sample

    input_t = sample["input"]
    target_t = sample["target"]
    target_mask_t = sample["target_mask"]

    assert input_t.shape == (input_steps, 1, 32, 32)
    assert target_t.shape == (forecast_steps, 1, 32, 32)
    assert target_mask_t.shape == (1, 1, 32, 32)  # collapsed over the sequence
    assert input_t.dtype == torch.float32
    assert target_t.dtype == torch.float32
    assert target_mask_t.dtype == torch.float32


def test_target_mask_collapses_over_sequence(fp_test_dataset: Path) -> None:
    """A cell that is NaN at *any* step (input or target) must be masked across
    every forecast step — a temporal discontinuity makes the trajectory there
    ill-defined, so the cell is not scored anywhere in the sequence."""
    input_steps, forecast_steps = 3, 2
    ds = SourceDataRandomSamplingDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        width=8,
        height=8,
        return_mask=True,
    )
    steps = input_steps + forecast_steps
    data = np.zeros((steps, 1, 8, 8), dtype=np.float32)
    data[1, 0, 2, 3] = np.nan  # NaN at one input step
    data[steps - 1, 0, 5, 5] = np.nan  # NaN at the last forecast step
    mask = ds._build_sample(data)["target_mask"].numpy()

    assert mask.shape == (1, 1, 8, 8)  # collapsed over the sequence
    assert mask[0, 0, 2, 3] == 0  # NaN at an input step masks the cell
    assert mask[0, 0, 5, 5] == 0  # NaN at a forecast step masks the cell
    assert mask[0, 0, 0, 0] == 1  # a cell valid at every step stays valid


def test_random_sampling_dataset_time_subset(fp_test_dataset: Path) -> None:
    """Test that subset correctly slices the Zarr store."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    ds = SourceDataRandomSamplingDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        input_steps=3,
        forecast_steps=2,
        subset={"time": (str(time_index[0]), str(time_index[49]))},
        epoch_size=10,
    )

    assert ds.max_t == 50
    assert len(ds) == 10


def test_random_sampling_dataset_forecast_steps_guard(fp_test_dataset: Path) -> None:
    """Test that instantiation with input_steps=0 raises ValueError."""
    with pytest.raises(ValueError, match="input_steps"):
        SourceDataRandomSamplingDataset(
            zarr_path=str(fp_test_dataset),
            standard_names=["rainfall_flux"],
            input_steps=0,
            forecast_steps=5,
        )

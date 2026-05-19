from pathlib import Path

import pandas as pd
import pytest
import torch
import xarray as xr

from mlcast.data.source_data_datasets import (
    SourceDataPrecomputedSamplingDataset,
    SourceDataRandomSamplingDataset,
)


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


def test_precomputed_sampling_dataset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Test that SourceDataPrecomputedSamplingDataset outputs the correct shape."""
    input_steps = 2
    forecast_steps = 1
    ds = SourceDataPrecomputedSamplingDataset(
        zarr_path=str(fp_test_dataset),
        csv_path=mock_csv,
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
    assert target_mask_t.shape == (forecast_steps, 1, 16, 16)
    assert isinstance(input_t, torch.Tensor)
    assert isinstance(target_t, torch.Tensor)
    assert isinstance(target_mask_t, torch.Tensor)


def test_precomputed_sampling_dataset_time_subset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Test that subset correctly filters CSV rows by time range."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    ds = SourceDataPrecomputedSamplingDataset(
        zarr_path=str(fp_test_dataset),
        csv_path=mock_csv,
        standard_names=["rainfall_flux"],
        input_steps=2,
        forecast_steps=1,
        subset={"time": (str(time_index[0]), str(time_index[8]))},
    )
    assert len(ds) == 2


def test_precomputed_sampling_dataset_forecast_steps_guard(fp_test_dataset: Path, mock_csv: str) -> None:
    """Test that instantiation with input_steps=0 raises ValueError."""
    with pytest.raises(ValueError, match="input_steps"):
        SourceDataPrecomputedSamplingDataset(
            zarr_path=str(fp_test_dataset),
            csv_path=mock_csv,
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
    assert target_mask_t.shape == (forecast_steps, 1, 32, 32)


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

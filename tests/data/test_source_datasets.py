import pandas as pd
import pytest
import torch

from mlcast.data.source_datasets import (
    SourceDataPrecomputedSamplingDataset,
    SourceDataRandomSamplingDataset,
)

ZARR_PATH = ".pytest_cache/italian_dataset_v0.1.0_100t.zarr"


@pytest.fixture
def mock_csv(tmp_path):
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


def test_precomputed_sampling_dataset(italian_dataset, mock_csv):
    """Test that SourceDataPrecomputedSamplingDataset outputs the correct shape."""
    ds = SourceDataPrecomputedSamplingDataset(
        zarr_path=ZARR_PATH,
        csv_path=mock_csv,
        standard_names=["rainfall_flux"],
        steps=3,
        width=16,
        height=16,
        return_mask=True,
    )

    assert len(ds) == 3
    sample = ds[0]

    assert "data" in sample
    assert "mask" in sample

    data = sample["data"]
    mask = sample["mask"]

    # Expected shape: (T, C, H, W) -> (3, 1, 16, 16)
    assert data.shape == (3, 1, 16, 16)
    assert mask.shape == (1, 1, 16, 16)
    assert isinstance(data, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_precomputed_sampling_dataset_time_slice(italian_dataset, mock_csv):
    """Test that time_slice correctly slices the CSV."""
    ds = SourceDataPrecomputedSamplingDataset(
        zarr_path=ZARR_PATH, csv_path=mock_csv, standard_names=["rainfall_flux"], steps=3, time_slice=slice(0, 2)
    )
    assert len(ds) == 2


def test_random_sampling_dataset(italian_dataset):
    """Test that SourceDataRandomSamplingDataset outputs the correct shape."""
    ds = SourceDataRandomSamplingDataset(
        zarr_path=ZARR_PATH,
        standard_names=["rainfall_flux"],
        steps=5,
        width=32,
        height=32,
        epoch_size=10,
        return_mask=True,
    )

    assert len(ds) == 10
    sample = ds[0]

    data = sample["data"]
    mask = sample["mask"]

    # Expected shape: (T, C, H, W) -> (5, 1, 32, 32)
    assert data.shape == (5, 1, 32, 32)
    assert mask.shape == (1, 1, 32, 32)


def test_random_sampling_dataset_time_slice(italian_dataset):
    """Test that time_slice correctly slices the Zarr store."""
    ds = SourceDataRandomSamplingDataset(
        zarr_path=ZARR_PATH, standard_names=["rainfall_flux"], steps=5, time_slice=slice(0, 50), epoch_size=10
    )

    assert ds.max_t == 50  # Since it was sliced to 50
    assert len(ds) == 10

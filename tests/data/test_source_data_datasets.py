from pathlib import Path

import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import Dataset

from mlcast.data.forecasting import ForecastingDataset
from mlcast.data.reconstruction import ReconstructionDataset
from mlcast.data.sequence import SourceDataPrecomputedSequenceDataset, SourceDataRandomSequenceDataset


class MockSequenceDataset(Dataset):
    def __init__(self, sequence_steps: int, num_samples: int = 2) -> None:
        self.sequence_steps = sequence_steps
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.arange(self.sequence_steps, dtype=torch.float32)[:, None, None, None].expand(-1, 1, 2, 2)


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


def test_precomputed_sequence_dataset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Precomputed sequence dataset should output normalized sequence tensors."""
    sequence_steps = 3
    ds = SourceDataPrecomputedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        csv_path=mock_csv,
        standard_names=["rainfall_flux"],
        sequence_steps=sequence_steps,
        width=16,
        height=16,
    )

    assert len(ds) == 3
    sample = ds[0]
    assert sample.shape == (sequence_steps, 1, 16, 16)
    assert sample.dtype == torch.float32


def test_precomputed_sequence_dataset_time_subset(fp_test_dataset: Path, mock_csv: str) -> None:
    """Subset should correctly filter CSV rows by time range."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    ds = SourceDataPrecomputedSequenceDataset(
        zarr_path=str(fp_test_dataset),
        csv_path=mock_csv,
        standard_names=["rainfall_flux"],
        sequence_steps=3,
        subset={"time": (str(time_index[0]), str(time_index[8]))},
    )
    assert len(ds) == 2


def test_precomputed_sequence_dataset_sequence_steps_guard(fp_test_dataset: Path, mock_csv: str) -> None:
    """Instantiation with sequence_steps=0 should raise ValueError."""
    with pytest.raises(ValueError, match="sequence_steps"):
        SourceDataPrecomputedSequenceDataset(
            zarr_path=str(fp_test_dataset),
            csv_path=mock_csv,
            standard_names=["rainfall_flux"],
            sequence_steps=0,
        )


def test_random_sequence_dataset(fp_test_dataset: Path) -> None:
    """Random sequence dataset should output normalized sequence tensors."""
    sequence_steps = 5
    ds = SourceDataRandomSequenceDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        sequence_steps=sequence_steps,
        width=32,
        height=32,
        epoch_size=10,
    )

    assert len(ds) == 10
    sample = ds[0]
    assert sample.shape == (sequence_steps, 1, 32, 32)
    assert sample.dtype == torch.float32


def test_random_sequence_dataset_time_subset(fp_test_dataset: Path) -> None:
    """Subset should correctly slice the Zarr store."""
    zarr_ds = xr.open_zarr(str(fp_test_dataset))
    time_index = zarr_ds.indexes["time"]
    ds = SourceDataRandomSequenceDataset(
        zarr_path=str(fp_test_dataset),
        standard_names=["rainfall_flux"],
        sequence_steps=5,
        subset={"time": (str(time_index[0]), str(time_index[49]))},
        epoch_size=10,
    )

    assert ds.max_t == 50
    assert len(ds) == 10


def test_forecasting_dataset_splits_sequence_and_derives_mask() -> None:
    """ForecastingDataset should split one sequence into input and target tensors."""
    base_dataset = MockSequenceDataset(sequence_steps=6)
    dataset = ForecastingDataset(base_dataset, input_steps=2, forecast_steps=4, return_mask=True)

    sample = dataset[0]
    assert sample["input"].shape == (2, 1, 2, 2)
    assert sample["target"].shape == (4, 1, 2, 2)
    assert sample["target_mask"].shape == (4, 1, 2, 2)
    assert torch.all(sample["target_mask"] == 1.0)


def test_reconstruction_dataset_creates_overlapping_windows() -> None:
    """ReconstructionDataset should expose all overlapping windows."""
    base_dataset = MockSequenceDataset(sequence_steps=5, num_samples=2)
    dataset = ReconstructionDataset(base_dataset, input_steps=3)

    assert len(dataset) == 6
    first_window = dataset[0]
    second_window = dataset[1]
    assert first_window.shape == (3, 1, 2, 2)
    assert torch.equal(first_window[:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(second_window[:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))

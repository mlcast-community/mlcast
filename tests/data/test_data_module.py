import functools
from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader, Dataset

from mlcast.data.source_data_datamodule import SourceDataDataModule


class MockDataset(Dataset):
    """Minimal dataset mock.

    ``__len__`` returns the number of time steps covered by ``time_slice``
    so that dataloader batch-count assertions work correctly.
    """

    def __init__(self, zarr_path: str, time_slice: slice | None = None, augment: bool = False, **kwargs) -> None:
        self.zarr_path = zarr_path
        self.time_slice = time_slice
        self.augment = augment
        self.kwargs = kwargs

    def __len__(self) -> int:
        if self.time_slice is not None:
            return self.time_slice.stop - self.time_slice.start
        return 0

    def __getitem__(self, idx: int) -> dict:
        return {"data": idx}


def _mock_zarr(n_time: int) -> MagicMock:
    """Return a mock xr.Dataset with a given time dimension size."""
    mock_ds = MagicMock()
    mock_ds.sizes = {"time": n_time}
    return mock_ds


def test_data_module_splits() -> None:
    """Test DataModule chronological split boundaries.

    Uses 100 time steps, train_ratio=0.5, val_ratio=0.2:
      train_end = int(100 * 0.5)      = 50
      val_end   = 50 + int(100 * 0.2) = 70
      test      = 70 to 100
    """
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr", foo="bar")

    dm = SourceDataDataModule(dataset_factory=dataset_factory, train_ratio=0.5, val_ratio=0.2, batch_size=2)

    with patch("mlcast.data.source_data_datamodule.xr.open_zarr", return_value=_mock_zarr(100)):
        dm.setup(stage="fit")

    assert dm.train_dataset.time_slice == slice(0, 50)
    assert dm.train_dataset.augment is True
    assert dm.train_dataset.kwargs["foo"] == "bar"

    assert dm.val_dataset.time_slice == slice(50, 70)
    assert dm.val_dataset.augment is False

    assert dm.test_dataset.time_slice == slice(70, 100)
    assert dm.test_dataset.augment is False

    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, DataLoader)
    assert train_dl.batch_size == 2


def test_data_module_invalid_dataset() -> None:
    """Ensure DataModule raises if zarr_path is not accessible via the factory."""

    class _NoZarrPathFactory:
        def __call__(self, **kwargs) -> Dataset:
            return MagicMock(spec=Dataset)

    dm = SourceDataDataModule(dataset_factory=_NoZarrPathFactory())

    with pytest.raises((AttributeError, KeyError)):
        dm.setup()


def test_data_module_split_lengths_and_batches() -> None:
    """Test that dataset lengths and dataloader batch counts are correct after splitting.

    Uses 240 time steps with a 1/2, 1/3, 1/6 train/val/test split and
    batch_size=10, chosen so all splits divide evenly and expected batch
    counts are easy to verify without rounding.

    Split boundaries (computed independently per split to avoid float accumulation):
      train_end = int(240 * 1/2)       = 120
      val_end   = 120 + int(240 * 1/3) = 120 + 80 = 200
      test      = 240 - 200            = 40

    Expected dataset lengths and batch counts at batch_size=10:
      train : 120 samples -> 12 batches
      val   :  80 samples ->  8 batches
      test  :  40 samples ->  4 batches
    """
    n_time = 240
    batch_size = 10
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        train_ratio=1 / 2,
        val_ratio=1 / 3,
        batch_size=batch_size,
    )

    with patch("mlcast.data.source_data_datamodule.xr.open_zarr", return_value=_mock_zarr(n_time)):
        dm.setup()

    assert len(dm.train_dataset) == 120, "train split should cover timesteps 0–120"
    assert len(dm.val_dataset) == 80, "val split should cover timesteps 120–200"
    assert len(dm.test_dataset) == 40, "test split should cover timesteps 200–240"

    assert len(dm.train_dataloader()) == 12, "120 samples / batch_size 10 = 12 batches"
    assert len(dm.val_dataloader()) == 8, "80 samples / batch_size 10 = 8 batches"
    assert len(dm.test_dataloader()) == 4, "40 samples / batch_size 10 = 4 batches"

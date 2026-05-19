import functools
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from torch.utils.data import DataLoader, Dataset

from mlcast.data.source_data_datamodule import SourceDataDataModule
from mlcast.data.splits import splitting_uses_fractions, splitting_uses_tuple_ranges, validate_splits


class MockDataset(Dataset):
    """Minimal dataset mock that records how it was constructed.

    ``__len__`` returns a fixed size so that dataloader batch-count assertions
    work correctly.
    """

    def __init__(
        self,
        zarr_path: str,
        subset: dict | None = None,
        augment: bool = False,
        epoch_size: int = 100,
        **kwargs,
    ) -> None:
        self.zarr_path = zarr_path
        self.subset = subset
        self.augment = augment
        self.epoch_size = epoch_size
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        return {"data": idx}


def _mock_zarr(time_index: pd.DatetimeIndex) -> MagicMock:
    """Return a mock xr.Dataset with a given pandas DatetimeIndex for time."""
    mock_ds = MagicMock()
    mock_ds.indexes = {"time": time_index}
    return mock_ds


def _make_time_index(n: int, start: str = "2016-01-01", freq: str = "10min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def test_validate_splits_ratio_mode() -> None:
    validate_splits({"time": {"train": 0.7, "val": 0.15}})
    validate_splits({"time": {"train": 0.7, "val": 0.15, "test": 0.15}})
    validate_splits({"time": {"train": 0.7, "val": 0.15, "test": None}})


def test_validate_splits_datetime_mode() -> None:
    validate_splits(
        {"time": {"train": ("2016-01-01", "2021-12-31"), "val": ("2022-01-01", "2023-12-31"), "test": None}}
    )


def test_validate_splits_missing_train() -> None:
    with pytest.raises(ValueError, match="must contain 'train'"):
        validate_splits({"time": {"val": 0.2}})


def test_validate_splits_ratio_exceeds_one() -> None:
    with pytest.raises(ValueError, match="sum to"):
        validate_splits({"time": {"train": 0.8, "val": 0.3}})


def test_validate_splits_ratio_requires_float_like_values() -> None:
    with pytest.raises(ValueError, match="float-like"):
        validate_splits({"time": {"train": "0.8", "val": 0.2}})

    with pytest.raises(ValueError, match="float-like"):
        validate_splits({"time": {"train": 0.8, "val": 0.1, "test": "0.1"}})


def test_validate_splits_warns_when_fraction_sum_is_not_one() -> None:
    with patch("mlcast.data.splits.logger.warning") as mock_warning:
        validate_splits({"time": {"train": 0.7, "val": 0.15}})

    mock_warning.assert_called_once()


def test_validate_splits_mixed_mode() -> None:
    with pytest.raises(ValueError, match="mix"):
        validate_splits({"time": {"train": 0.7, "val": ("2022-01-01", "2023-12-31")}})


def test_validate_splits_datetime_missing_test() -> None:
    with pytest.raises(ValueError, match="must contain 'test'"):
        validate_splits({"time": {"train": ("2016-01-01", "2021-12-31"), "val": ("2022-01-01", "2023-12-31")}})


def test_validate_splits_unknown_coord() -> None:
    with pytest.raises(ValueError, match="Unknown coordinate"):
        validate_splits({"space": {"train": 0.7, "val": 0.2}})


def test_splitting_mode_helpers_require_consistent_values() -> None:
    assert splitting_uses_fractions({"train": 0.7, "val": 0.2, "test": None})
    assert not splitting_uses_fractions({"train": 0.7, "val": ("2022-01-01", "2022-12-31")})
    assert splitting_uses_tuple_ranges(
        {"train": ("2016-01-01", "2021-12-31"), "val": ("2022-01-01", "2023-12-31"), "test": None}
    )
    assert not splitting_uses_tuple_ranges({"train": object(), "val": object()})


def test_data_module_ratio_splits() -> None:
    """DataModule ratio mode passes correct time subsets to the factory."""
    n = 100
    time_index = _make_time_index(n)
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr", foo="bar")

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory, splits={"time": {"train": 0.5, "val": 0.2, "test": 0.3}}, batch_size=2
    )

    with patch("mlcast.data.splits.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup(stage="fit")

    assert dm.train_dataset.augment is True
    assert dm.train_dataset.kwargs["foo"] == "bar"
    train_start, train_end = dm.train_dataset.subset["time"]
    val_start, val_end = dm.val_dataset.subset["time"]
    test_start, test_end = dm.test_dataset.subset["time"]

    assert train_start == str(time_index[0])
    assert train_end == str(time_index[49])
    assert val_start == str(time_index[50])
    assert val_end == str(time_index[69])
    assert test_start == str(time_index[70])
    assert test_end == str(time_index[99])

    assert dm.val_dataset.augment is False
    assert dm.test_dataset.augment is False

    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, DataLoader)
    assert train_dl.batch_size == 2


def test_data_module_invalid_dataset() -> None:
    """Ensure DataModule raises if zarr_path is not accessible via the factory."""

    class _NoZarrPathFactory:
        def __call__(self, **kwargs) -> Dataset:
            return MagicMock(spec=Dataset)

    dm = SourceDataDataModule(dataset_factory=_NoZarrPathFactory(), splits={"time": {"train": 0.7, "val": 0.15}})

    with pytest.raises((AttributeError, KeyError)):
        dm.setup()


def test_data_module_fraction_splits_without_test_do_not_create_test_dataset() -> None:
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")
    time_index = _make_time_index(100)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 0.5, "val": 0.2}},
        batch_size=2,
    )

    with patch("mlcast.data.splits.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is None


def test_data_module_split_lengths_and_batches() -> None:
    """Test that dataset lengths and dataloader batch counts are correct after splitting.

    Dataloader batch counts are correct after splitting.
    """
    n_time = 240
    batch_size = 10
    time_index = _make_time_index(n_time)
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr", epoch_size=10)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 1 / 2, "val": 1 / 3, "test": 1 / 6}},
        batch_size=batch_size,
    )

    with patch("mlcast.data.splits.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup()

    assert len(dm.train_dataloader()) == 1
    assert len(dm.val_dataloader()) == 1
    assert len(dm.test_dataloader()) == 1


def test_data_module_datetime_splits() -> None:
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={
            "time": {
                "train": ("2016-01-01", "2021-12-31"),
                "val": ("2022-01-01", "2023-12-31"),
                "test": None,
            }
        },
        batch_size=4,
    )

    dm.setup()

    assert dm.train_dataset.subset == {"time": ("2016-01-01", "2021-12-31")}
    assert dm.val_dataset.subset == {"time": ("2022-01-01", "2023-12-31")}
    assert dm.test_dataset is None


def test_data_module_fraction_test_split_uses_explicit_fraction() -> None:
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")
    time_index = _make_time_index(100)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 0.5, "val": 0.2, "test": 0.1}},
        batch_size=2,
    )

    with patch("mlcast.data.splits.xr.open_zarr", return_value=_mock_zarr(time_index)):
        dm.setup()

    assert dm.test_dataset is not None
    test_start, test_end = dm.test_dataset.subset["time"]
    assert test_start == str(time_index[70])
    assert test_end == str(time_index[79])


def test_data_module_logs_split_summary() -> None:
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")
    time_index = _make_time_index(100)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        splits={"time": {"train": 0.5, "val": 0.2, "test": 0.1}},
        batch_size=2,
    )

    with (
        patch("mlcast.data.splits.xr.open_zarr", return_value=_mock_zarr(time_index)),
        patch("mlcast.data.source_data_datamodule.logger.info") as mock_info,
    ):
        dm.setup()

    assert mock_info.call_count == 4


def test_data_module_unsupported_split_mode() -> None:
    dataset_factory = functools.partial(MockDataset, zarr_path="mock.zarr")
    dm = SourceDataDataModule(dataset_factory=dataset_factory, splits={"time": {"train": 0.7, "val": 0.15}})

    dm.splits = {"time": {"train": object(), "val": object()}}

    with pytest.raises(NotImplementedError, match="Unsupported split mode"):
        dm.setup()

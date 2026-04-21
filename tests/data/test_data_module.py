import functools
from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader, Dataset

from mlcast.data.source_data_module import SourceDataDataModule


class MockDataset(Dataset):
    def __init__(self, time_slice=None, augment=False, **kwargs):
        self.time_slice = time_slice
        self.augment = augment
        self.kwargs = kwargs

    def __len__(self):
        if self.time_slice:
            return self.time_slice.stop - self.time_slice.start
        return 100

    def __getitem__(self, idx):
        return {"data": idx}


class MockPrecomputedDataset(MockDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulate having coords attribute
        self.coords = [1] * 100


class MockRandomDataset(MockDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulate having ds.time attribute
        self.ds = MagicMock()
        self.ds.time.size = 200


def test_data_module_splits_precomputed():
    """Test DataModule chronological splits using a mock dataset factory."""
    # Use functools.partial to simulate fdl.Partial
    dataset_factory = functools.partial(MockPrecomputedDataset, foo="bar")

    dm = SourceDataDataModule(dataset_factory=dataset_factory, train_ratio=0.5, val_ratio=0.2, batch_size=2)

    dm.setup(stage="fit")

    # 100 items total * 0.5 = 50 train
    assert dm.train_dataset.time_slice == slice(0, 50)
    assert dm.train_dataset.augment is True
    assert dm.train_dataset.kwargs["foo"] == "bar"

    # 100 * 0.7 = 70. Val is 50 to 70.
    assert dm.val_dataset.time_slice == slice(50, 70)
    assert dm.val_dataset.augment is False

    # 70 to 100.
    assert dm.test_dataset.time_slice == slice(70, 100)
    assert dm.test_dataset.augment is False

    train_dl = dm.train_dataloader()
    assert isinstance(train_dl, DataLoader)
    assert train_dl.batch_size == 2


def test_data_module_splits_random():
    """Test DataModule chronological splits using a mock random dataset factory."""
    dataset_factory = functools.partial(MockRandomDataset)

    dm = SourceDataDataModule(
        dataset_factory=dataset_factory,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    dm.setup()

    # 200 items total * 0.8 = 160 train
    assert dm.train_dataset.time_slice == slice(0, 160)

    # 200 * 0.9 = 180. Val is 160 to 180.
    assert dm.val_dataset.time_slice == slice(160, 180)

    # 180 to 200.
    assert dm.test_dataset.time_slice == slice(180, 200)


def test_data_module_invalid_dataset():
    """Ensure DataModule raises an error if the dataset lacks length attributes."""
    dataset_factory = functools.partial(MockDataset)

    dm = SourceDataDataModule(dataset_factory=dataset_factory)

    with pytest.raises(ValueError, match="Dataset must have 'coords' or 'ds.time'"):
        dm.setup()

"""PyTorch Lightning data module for spatio-temporal datasets.

Handles train/val/test splitting and DataLoader creation from a single
Zarr store and CSV coordinate file produced by mlcast-dataset-sampler.
"""

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .source_datasets import SourceDataPrecomputedSamplingDataset


class SourceDataDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for spatio-temporal datasets.

    Handles train/val/test splitting and DataLoader creation from a single
    Zarr store and CSV coordinate file.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with crop coordinates.
    standard_names : list of str
        Names of the standard CF variables to load from the Zarr store.
    steps : int
        Number of timesteps per sample.
    train_ratio : float, optional
        Fraction of data used for training. Default is ``0.7``.
    val_ratio : float, optional
        Fraction of data used for validation. Default is ``0.15``.
    return_mask : bool, optional
        Whether to return NaN masks. Default is ``False``.
    deterministic : bool, optional
        Whether to use fixed random seeds. Default is ``False``.
    augment : bool, optional
        Whether to apply data augmentation (training set only). Default is
        ``True``.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    time_depth : int, optional
        Number of timesteps in the sampled window. Default is ``24``.
    **dataloader_kwargs
        Additional keyword arguments forwarded to ``DataLoader`` (e.g.
        ``batch_size``, ``num_workers``, ``pin_memory``).
    """

    def __init__(
        self,
        zarr_path,
        csv_path,
        standard_names,
        steps,
        train_ratio=0.7,
        val_ratio=0.15,
        return_mask=False,
        deterministic=False,
        augment=True,
        width=256,
        height=256,
        time_depth=24,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.csv_path = csv_path
        self.standard_names = standard_names
        self.steps = steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dataloader_kwargs = dataloader_kwargs
        self.return_mask = return_mask
        self.deterministic = deterministic
        self.augment = augment
        self.width = width
        self.height = height
        self.time_depth = time_depth

    def setup(self, stage=None):
        """Create train, validation, and test datasets.

        Splits are chronological: the first ``train_ratio`` fraction is used
        for training, the next ``val_ratio`` for validation, and the rest for
        testing. Augmentation is only applied to the training set.
        """
        coords = pd.read_csv(self.csv_path).sort_values("t")
        n = len(coords)

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        common_kwargs = dict(
            zarr_path=self.zarr_path,
            csv_path=self.csv_path,
            standard_names=self.standard_names,
            steps=self.steps,
            return_mask=self.return_mask,
            deterministic=self.deterministic,
            width=self.width,
            height=self.height,
            time_depth=self.time_depth,
        )

        self.train_dataset = SourceDataPrecomputedSamplingDataset(
            **common_kwargs,
            augment=self.augment,
            time_slice=slice(0, train_end),
        )
        self.val_dataset = SourceDataPrecomputedSamplingDataset(
            **common_kwargs,
            augment=False,
            time_slice=slice(train_end, val_end),
        )
        self.test_dataset = SourceDataPrecomputedSamplingDataset(
            **common_kwargs,
            augment=False,
            time_slice=slice(val_end, n),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)

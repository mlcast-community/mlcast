"""PyTorch dataset for loading radar datacubes from Zarr stores.

Loads spatio-temporal datacubes using pre-sampled coordinates from a CSV
file produced by mlcast-dataset-sampler.
"""

import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from mlcast.utils import rainrate_to_normalized


class SampledRadarDataset(Dataset):
    """PyTorch dataset that loads radar datacubes from a Zarr store using
    pre-sampled spatial-temporal coordinates from a CSV file.

    Each sample is a spatio-temporal datacube of shape ``(T, 1, H, W)``
    converted from rain rate to normalized reflectivity.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with columns ``(t, x, y)`` specifying the
        top-left corner of each datacube.
    variable_name : str
        Name of the variable to load from the Zarr store (e.g. ``'RR'``,
        ``'LBMR'``).
    steps : int
        Number of timesteps to extract per sample.
    return_mask : bool, optional
        If ``True``, also return a spatial NaN mask. Default is ``False``.
    deterministic : bool, optional
        If ``True``, use a fixed random seed (42) for reproducibility.
        Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations (rotation, flips).
        Default is ``False``.
    indices : sequence of int or None, optional
        Subset of row indices to use from the CSV. If ``None``, use all rows.
        Default is ``None``.
    width : int, optional
        Spatial width of each datacube. Default is ``256``.
    height : int, optional
        Spatial height of each datacube. Default is ``256``.
    time_depth : int, optional
        Number of timesteps in the sampled window. Default is ``24``.
    """

    def __init__(
        self,
        zarr_path: str,
        csv_path: str,
        variable_name: str,
        steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        indices=None,
        width: int = 256,
        height: int = 256,
        time_depth: int = 24,
    ):
        self.coords = pd.read_csv(csv_path).sort_values("t")
        if indices is not None:
            self.coords = self.coords.iloc[list(indices)].reset_index(drop=True)
        self.zg = xr.open_zarr(zarr_path)
        self.data_var = self.zg[variable_name]
        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))
        self.return_mask = return_mask
        self.augment = augment

        self.w = width
        self.h = height
        self.dt = time_depth
        self.steps = steps

        if self.steps > self.dt:
            print(f"Warning: requested steps ({self.steps}) > sampled time window ({self.dt})")

    def __len__(self):
        return len(self.coords)

    def shape(self):
        return (len(self.coords), self.steps, 1, self.w, self.h)

    def _apply_augmentations(
        self, *tensors, rotate_prob: float = 0.5, hflip_prob: float = 0.5, vflip_prob: float = 0.5
    ):
        """Apply random spatial augmentations consistently to all input tensors."""
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]

        if self.rng.random() < hflip_prob:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]

        if self.rng.random() < vflip_prob:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]

        tensors = [t.contiguous() for t in tensors]
        return tensors[0] if len(tensors) == 1 else tuple(tensors)

    def __getitem__(self, idx: int):
        """Load and return a single datacube sample.

        Returns
        -------
        sample : dict of str to torch.Tensor
            Dictionary with key ``'data'`` containing a tensor of shape
            ``(T, 1, H, W)``. If ``return_mask`` is ``True``, also contains
            ``'mask'`` of shape ``(1, 1, H, W)``.
        """
        t0, x0, y0 = self.coords.iloc[idx]

        x_slice = slice(x0, x0 + self.w)
        y_slice = slice(y0, y0 + self.h)

        if self.steps < self.dt:
            t_start = self.rng.integers(t0, t0 + self.dt - self.steps + 1)
        else:
            t_start = t0
        t_slice = slice(t_start, t_start + self.steps)

        data = rainrate_to_normalized(self.data_var[t_slice, x_slice, y_slice])

        if self.return_mask:
            mask = (~(np.isnan(data).any(axis=0, keepdims=True))).astype(np.float32)

        data = np.nan_to_num(data, nan=-1.0)

        data = torch.from_numpy(data[:, np.newaxis, :, :])
        if self.return_mask:
            mask = torch.from_numpy(mask.values[:, np.newaxis, :, :])

        if self.augment:
            if self.return_mask:
                data, mask = self._apply_augmentations(data, mask)
            else:
                data = self._apply_augmentations(data)

        if self.return_mask:
            return {"data": data, "mask": mask}
        else:
            return {"data": data}

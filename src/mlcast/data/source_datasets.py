"""PyTorch datasets for loading spatio-temporal data from Zarr stores.

Provides pre-computed sampling and (soon) random sampling datasets.
"""

import time
import warnings

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import torch
import xarray as xr
from beartype import beartype
from jaxtyping import jaxtyped
from torch.utils.data import Dataset

from mlcast.data.normalization import NORMALIZATION_REGISTRY


class SourceDataPrecomputedSamplingDataset(Dataset):
    """PyTorch dataset that loads spatio-temporal data from a Zarr store using
    pre-sampled spatial-temporal coordinates from a CSV file.

    Each sample is a spatio-temporal crop of shape ``(T, C, H, W)``
    converted to normalized reflectivity.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with columns ``(t, x, y)`` specifying the
        top-left corner of each crop.
    standard_names : list of str
        List of CF standard names of variables to load (e.g., ``["rainfall_rate"]``).
    steps : int
        Number of timesteps to extract per sample.
    return_mask : bool, optional
        If ``True``, also return a spatial NaN mask. Default is ``False``.
    deterministic : bool, optional
        If ``True``, use a fixed random seed (42) for reproducibility. Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations (rotation, flips). Default is ``False``.
    time_slice : slice or None, optional
        Subset of row indices to use from the CSV for train/val splitting.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    time_depth : int, optional
        Number of timesteps in the sampled window. Default is ``24``.
    """

    def __init__(
        self,
        zarr_path: str,
        csv_path: str,
        standard_names: list[str],
        steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        time_slice: slice | None = None,
        width: int = 256,
        height: int = 256,
        time_depth: int = 24,
    ) -> None:
        self.coords = pd.read_csv(csv_path).sort_values("t")
        if time_slice is not None:
            self.coords = self.coords.iloc[time_slice].reset_index(drop=True)

        self.ds = xr.open_zarr(zarr_path)
        self.standard_names = standard_names
        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))
        self.return_mask = return_mask
        self.augment = augment

        self.w = width
        self.h = height
        self.dt = time_depth
        self.steps = steps

        da_first_var = self.ds.cf[self.standard_names[0]]
        self.t_dim = da_first_var.cf["time"].dims[0]

        if "Y" in da_first_var.cf.axes:
            self.y_dim = da_first_var.cf.axes["Y"][0]
        else:
            warnings.warn(
                "cf_xarray could not find 'Y' axis via CF conventions. Falling back to dimension named 'y'.",
                stacklevel=2,
            )
            self.y_dim = "y"

        if "X" in da_first_var.cf.axes:
            self.x_dim = da_first_var.cf.axes["X"][0]
        else:
            warnings.warn(
                "cf_xarray could not find 'X' axis via CF conventions. Falling back to dimension named 'x'.",
                stacklevel=2,
            )
            self.x_dim = "x"

        if self.steps > self.dt:
            print(f"Warning: requested steps ({self.steps}) > sampled time window ({self.dt})")

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        length : int
            Number of samples.
        """
        return len(self.coords)

    def _apply_augmentations(
        self, *tensors: torch.Tensor, rotate_prob: float = 0.5, hflip_prob: float = 0.5, vflip_prob: float = 0.5
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply random spatial augmentations consistently to all input tensors."""
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)
            tensors = tuple(torch.rot90(t, int(k), dims=[-2, -1]) for t in tensors)

        if self.rng.random() < hflip_prob:
            tensors = tuple(torch.flip(t, dims=[-1]) for t in tensors)

        if self.rng.random() < vflip_prob:
            tensors = tuple(torch.flip(t, dims=[-2]) for t in tensors)

        tensors = tuple(t.contiguous() for t in tensors)
        return tensors[0] if len(tensors) == 1 else tensors

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return a single crop sample.

        Returns
        -------
        sample : dict of str to torch.Tensor
            Dictionary with key ``'data'`` containing a tensor of shape
            ``(T, C, H, W)``. If ``return_mask`` is ``True``, also contains
            ``'mask'`` of shape ``(1, 1, H, W)``.
        """
        t0, x0, y0 = self.coords.iloc[idx]

        x_slice = slice(int(x0), int(x0) + self.w)
        y_slice = slice(int(y0), int(y0) + self.h)

        if self.steps < self.dt:
            t_start = self.rng.integers(t0, t0 + self.dt - self.steps + 1)
        else:
            t_start = t0
        t_slice = slice(int(t_start), int(t_start) + self.steps)

        channels = []
        masks = []
        for std_name in self.standard_names:
            da_var = self.ds.cf[std_name].isel({self.t_dim: t_slice, self.x_dim: x_slice, self.y_dim: y_slice})
            var_data = da_var.values

            norm_func = NORMALIZATION_REGISTRY[std_name]
            norm_data = norm_func(var_data)
            channels.append(norm_data)

            if self.return_mask:
                masks.append((~(np.isnan(norm_data).any(axis=0, keepdims=True))).astype(np.float32))

        # Stack along channel dimension: (C, T, H, W) -> (T, C, H, W)
        data = np.stack(channels, axis=0)
        data = np.swapaxes(data, 0, 1)
        data = np.nan_to_num(data, nan=-1.0)
        data_t = torch.from_numpy(data)

        if self.return_mask:
            # Combine masks across channels: valid only if all channels are valid
            mask = np.stack(masks, axis=0).min(axis=0)  # shape (1, H, W)
            mask_t = torch.from_numpy(mask[np.newaxis, ...])  # shape (1, 1, H, W)

            if self.augment:
                augmented = self._apply_augmentations(data_t, mask_t)
                assert isinstance(augmented, tuple)
                data_t, mask_t = augmented
            return {"data": data_t, "mask": mask_t}
        else:
            if self.augment:
                augmented = self._apply_augmentations(data_t)
                assert isinstance(augmented, torch.Tensor)
                data_t = augmented
            return {"data": data_t}

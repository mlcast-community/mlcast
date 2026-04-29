"""PyTorch datasets for loading spatio-temporal data from Zarr stores.

Provides pre-computed sampling and (soon) random sampling datasets.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, TypedDict

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import torch
import xarray as xr
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch.utils.data import Dataset

from mlcast.data.normalization import NORMALIZATION_REGISTRY


class DatasetSample(TypedDict, total=False):
    """Typed dictionary returned by dataset ``__getitem__``.

    Keys
    ----
    input : Float[torch.Tensor, "input_steps channels height width"]
        Past frames fed to the network as input.
    target : Float[torch.Tensor, "forecast_steps channels height width"]
        Future frames the network should predict.
    target_mask : Float[torch.Tensor, "forecast_steps channels height width"]
        Per-timestep, per-channel validity mask for the target (1 = valid,
        0 = NaN in original data). Only present when ``return_mask=True``.
    """

    input: Float[torch.Tensor, "input_steps channels height width"]
    target: Float[torch.Tensor, "forecast_steps channels height width"]
    target_mask: Float[torch.Tensor, "forecast_steps channels height width"]


def _detect_axes(ds: xr.Dataset, standard_name: str) -> tuple[str, str, str]:
    """Detect CF axis dimension names for a variable in an xarray Dataset.

    Falls back to dimension names ``'y'`` / ``'x'`` when CF conventions do not
    identify the axis, emitting a :mod:`warnings` warning in each case.

    Parameters
    ----------
    ds : xr.Dataset
        An open xarray Dataset with CF conventions.
    standard_name : str
        A CF standard name present in ``ds``, used to look up the variable.

    Returns
    -------
    t_dim : str
        Dimension name for the time axis.
    y_dim : str
        Dimension name for the Y (latitude) axis.
    x_dim : str
        Dimension name for the X (longitude) axis.
    """
    da = ds.cf[standard_name]
    t_dim = da.cf["time"].dims[0]

    if "Y" in da.cf.axes:
        y_dim = da.cf.axes["Y"][0]
    else:
        warnings.warn(
            "cf_xarray could not find 'Y' axis via CF conventions. Falling back to dimension named 'y'.",
            stacklevel=3,
        )
        y_dim = "y"

    if "X" in da.cf.axes:
        x_dim = da.cf.axes["X"][0]
    else:
        warnings.warn(
            "cf_xarray could not find 'X' axis via CF conventions. Falling back to dimension named 'x'.",
            stacklevel=3,
        )
        x_dim = "x"

    return t_dim, y_dim, x_dim


class SourceDataDatasetBase(Dataset, ABC):
    """Abstract base class for mlcast Zarr-backed spatio-temporal datasets.

    Subclasses must implement :meth:`__len__` and :meth:`__getitem__`.
    All common initialisation, Zarr access, CF-axis resolution, augmentation,
    and the ``steps`` property live here.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    standard_names : list of str
        List of CF standard names of variables to load.
    input_steps : int
        Number of past timesteps fed to the network as input.
    forecast_steps : int
        Number of future timesteps the network should predict.
    return_mask : bool, optional
        If ``True``, also return a per-timestep validity mask for the target.
        Default is ``False``.
    deterministic : bool, optional
        If ``True``, use a fixed random seed (42). Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations. Default is ``False``.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    storage_options : dict or None, optional
        Options forwarded to ``xr.open_zarr``. Default is ``None``.
    """

    def __init__(
        self,
        zarr_path: str,
        standard_names: list[str],
        input_steps: int,
        forecast_steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        width: int = 256,
        height: int = 256,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        if input_steps < 1:
            raise ValueError(f"input_steps ({input_steps}) must be at least 1.")
        if forecast_steps < 1:
            raise ValueError(f"forecast_steps ({forecast_steps}) must be at least 1.")

        self.storage_options = storage_options
        self._zarr_path = zarr_path
        self._ds: xr.Dataset | None = None
        self.standard_names = standard_names
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.return_mask = return_mask
        self.augment = augment
        self.w = width
        self.h = height
        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))

        self._validate_standard_names()
        self.t_dim, self.y_dim, self.x_dim = _detect_axes(self.ds, self.standard_names[0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        """Total number of timesteps per sample (``input_steps + forecast_steps``).

        Returns
        -------
        steps : int
            ``input_steps + forecast_steps``.
        """
        return self.input_steps + self.forecast_steps

    @property
    def ds(self) -> xr.Dataset:
        """Open and cache the Zarr-backed xarray Dataset for this worker.

        The store is opened lazily on first access within each process. This
        avoids pickling live asyncio connections across DataLoader worker
        boundaries, which would cause ``RuntimeError: Future attached to a
        different loop``.

        Returns
        -------
        ds : xr.Dataset
            The opened (and optionally time-sliced) xarray Dataset.
        """
        if self._ds is None:
            ds = xr.open_zarr(self._zarr_path, storage_options=self.storage_options)
            if self._time_slice is not None:
                ds = ds.isel(time=self._time_slice)
            self._ds = ds
        return self._ds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_standard_names(self) -> None:
        """Check that every requested CF standard name exists in the Zarr store.

        Raises
        ------
        ValueError
            If a requested standard name is not found.
        """
        for std_name in self.standard_names:
            try:
                _ = self.ds.cf[std_name]
            except KeyError as e:
                if hasattr(self.ds.cf, "standard_names"):
                    available_cf_names = list(self.ds.cf.standard_names.keys())
                else:
                    available_cf_names = []

                if not available_cf_names:
                    msg = (
                        f"Requested CF standard_name '{std_name}' not found. "
                        "In fact, this dataset has NO variables with a 'standard_name' CF attribute. "
                        "Please ensure the Zarr dataset is properly formatted with CF conventions."
                    )
                else:
                    msg = (
                        f"Requested CF standard_name '{std_name}' not found in the dataset.\n"
                        f"Available CF standard names: {available_cf_names}\n"
                        f"\nHint: You can change the requested variables via the CLI using:\n"
                        f"  --config \"fiddler:set_variables(standard_names=['<correct_name>'])\""
                    )
                raise ValueError(msg) from e

    def _apply_augmentations(
        self, *tensors: torch.Tensor, rotate_prob: float = 0.5, hflip_prob: float = 0.5, vflip_prob: float = 0.5
    ) -> tuple[torch.Tensor, ...]:
        """Apply random spatial augmentations consistently to all input tensors."""
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)
            tensors = tuple(torch.rot90(t, int(k), dims=[-2, -1]) for t in tensors)

        if self.rng.random() < hflip_prob:
            tensors = tuple(torch.flip(t, dims=[-1]) for t in tensors)

        if self.rng.random() < vflip_prob:
            tensors = tuple(torch.flip(t, dims=[-2]) for t in tensors)

        return tuple(t.contiguous() for t in tensors)

    def _build_sample(self, data: np.ndarray) -> DatasetSample:
        """Convert a raw ``(T, C, H, W)`` numpy array into a :class:`DatasetSample`.

        Computes the target mask (before ``nan_to_num``), splits into input /
        target tensors along the time axis, applies augmentations if requested,
        and assembles the final dict.

        Parameters
        ----------
        data : np.ndarray
            Raw normalised array of shape ``(steps, C, H, W)`` — may contain
            NaNs where the original data was invalid.

        Returns
        -------
        sample : DatasetSample
            Dictionary with ``'input'`` and ``'target'`` tensors, and
            optionally ``'target_mask'`` if ``self.return_mask`` is ``True``.
        """
        # Capture target mask before NaNs are filled
        if self.return_mask:
            target_mask_t = torch.from_numpy((~np.isnan(data[self.input_steps :])).astype(np.float32))

        data = np.nan_to_num(data, nan=-1.0)
        data_t = torch.from_numpy(data)

        input_t = data_t[: self.input_steps]
        target_t = data_t[self.input_steps :]

        if self.augment:
            tensors = (input_t, target_t, target_mask_t) if self.return_mask else (input_t, target_t)
            augmented = self._apply_augmentations(*tensors)
            if self.return_mask:
                input_t, target_t, target_mask_t = augmented
            else:
                input_t, target_t = augmented

        sample = DatasetSample(input=input_t, target=target_t)
        if self.return_mask:
            sample["target_mask"] = target_mask_t
        return sample

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetSample: ...


class SourceDataPrecomputedSamplingDataset(SourceDataDatasetBase):
    """PyTorch dataset that loads spatio-temporal data from a Zarr store using
    pre-sampled spatial-temporal coordinates from a CSV file.

    Each sample is a spatio-temporal crop of shape ``(T, C, H, W)``
    converted to normalized data.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with columns ``(t, x, y)`` specifying the
        top-left corner of each crop.
    standard_names : list of str
        List of CF standard names of variables to load (e.g., ``["rainfall_rate"]``).
    input_steps : int
        Number of past timesteps fed to the network as input.
    forecast_steps : int
        Number of future timesteps the network should predict.
    return_mask : bool, optional
        If ``True``, also return a per-timestep validity mask for the target.
        Default is ``False``.
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
        input_steps: int,
        forecast_steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        time_slice: slice | None = None,
        width: int = 256,
        height: int = 256,
        time_depth: int = 24,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self._time_slice: slice | None = None  # required by base ds property before super().__init__ opens store
        super().__init__(
            zarr_path=zarr_path,
            standard_names=standard_names,
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            return_mask=return_mask,
            deterministic=deterministic,
            augment=augment,
            width=width,
            height=height,
            storage_options=storage_options,
        )

        self.coords = pd.read_csv(csv_path).sort_values("t")
        if time_slice is not None:
            self.coords = self.coords.iloc[time_slice].reset_index(drop=True)

        self.dt = time_depth

        if self.steps > self.dt:
            print(f"Warning: requested steps ({self.steps}) > sampled time window ({self.dt})")

        # Close the store: metadata has been extracted into plain attributes above.
        # Each DataLoader worker will reopen it via the `ds` property in its own
        # event loop, avoiding asyncio "Future attached to a different loop" errors.
        self._ds = None

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        length : int
            Number of samples.
        """
        return len(self.coords)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> DatasetSample:
        """Load and return a single crop sample.

        Returns
        -------
        sample : DatasetSample
            Dictionary with keys ``'input'`` of shape
            ``(input_steps, C, H, W)`` and ``'target'`` of shape
            ``(forecast_steps, C, H, W)``.  If ``return_mask`` is ``True``,
            also contains ``'target_mask'`` of shape
            ``(forecast_steps, C, H, W)`` with 1 where the original data was
            valid and 0 where it was NaN.
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
        for std_name in self.standard_names:
            da_var = self.ds.cf[std_name].isel({self.t_dim: t_slice, self.x_dim: x_slice, self.y_dim: y_slice})
            norm_func = NORMALIZATION_REGISTRY[std_name]
            channels.append(norm_func(da_var.values))

        data = np.swapaxes(np.stack(channels, axis=0), 0, 1)
        return self._build_sample(data)


class SourceDataRandomSamplingDataset(SourceDataDatasetBase):
    """PyTorch dataset that performs on-the-fly random spatial and temporal
    slicing of a Zarr store spatio-temporal data array.

    Each sample is a spatio-temporal crop of shape ``(T, C, H, W)``
    converted to normalized data.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    standard_names : list of str
        List of CF standard names of variables to load (e.g., ``["rainfall_rate"]``).
    input_steps : int
        Number of past timesteps fed to the network as input.
    forecast_steps : int
        Number of future timesteps the network should predict.
    return_mask : bool, optional
        If ``True``, also return a per-timestep validity mask for the target.
        Default is ``False``.
    deterministic : bool, optional
        If ``True``, use a fixed random seed (42) for reproducibility. Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations (rotation, flips). Default is ``False``.
    time_slice : slice or None, optional
        Subset of time indices to use for train/val splitting.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    epoch_size : int, optional
        Number of random samples to generate per epoch. Default is ``1000``.
    **kwargs : Any
        Ignored extra arguments (e.g. ``csv_path``) to allow drop-in replacement.
    """

    def __init__(
        self,
        zarr_path: str,
        standard_names: list[str],
        input_steps: int,
        forecast_steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        time_slice: slice | None = None,
        width: int = 256,
        height: int = 256,
        epoch_size: int = 1000,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._time_slice = time_slice  # required by base ds property before super().__init__ opens store
        super().__init__(
            zarr_path=zarr_path,
            standard_names=standard_names,
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            return_mask=return_mask,
            deterministic=deterministic,
            augment=augment,
            width=width,
            height=height,
            storage_options=storage_options,
        )

        self.epoch_size = epoch_size

        da_first_var = self.ds.cf[self.standard_names[0]]
        self.max_t = da_first_var.sizes[self.t_dim]
        self.max_y = da_first_var.sizes[self.y_dim]
        self.max_x = da_first_var.sizes[self.x_dim]

        if self.steps > self.max_t:
            raise ValueError(f"Requested steps ({self.steps}) > available time dimension ({self.max_t})")
        if self.h > self.max_y:
            raise ValueError(f"Requested height ({self.h}) > available Y dimension ({self.max_y})")
        if self.w > self.max_x:
            raise ValueError(f"Requested width ({self.w}) > available X dimension ({self.max_x})")

        # Close the store: metadata has been extracted into plain attributes above.
        # Each DataLoader worker will reopen it via the `ds` property in its own
        # event loop, avoiding asyncio "Future attached to a different loop" errors.
        self._ds = None

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        length : int
            Number of samples.
        """
        return self.epoch_size

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> DatasetSample:
        """Load and return a single randomly sampled datacube.

        Returns
        -------
        sample : DatasetSample
            Dictionary with keys ``'input'`` of shape
            ``(input_steps, C, H, W)`` and ``'target'`` of shape
            ``(forecast_steps, C, H, W)``.  If ``return_mask`` is ``True``,
            also contains ``'target_mask'`` of shape
            ``(forecast_steps, C, H, W)`` with 1 where the original data was
            valid and 0 where it was NaN.
        """
        t_start = self.rng.integers(0, self.max_t - self.steps + 1)
        y_start = self.rng.integers(0, self.max_y - self.h + 1)
        x_start = self.rng.integers(0, self.max_x - self.w + 1)

        t_slice = slice(int(t_start), int(t_start) + self.steps)
        y_slice = slice(int(y_start), int(y_start) + self.h)
        x_slice = slice(int(x_start), int(x_start) + self.w)

        channels = []
        for std_name in self.standard_names:
            da_var = self.ds.cf[std_name].isel({self.t_dim: t_slice, self.x_dim: x_slice, self.y_dim: y_slice})
            norm_func = NORMALIZATION_REGISTRY[std_name]
            channels.append(norm_func(da_var.values))

        data = np.swapaxes(np.stack(channels, axis=0), 0, 1)
        return self._build_sample(data)

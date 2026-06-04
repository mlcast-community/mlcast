"""Source-data sequence datasets built from Zarr stores.

These datasets are responsible for sampling normalized spatio-temporal
sequences directly from source datasets. They do not impose any forecasting or
reconstruction task structure on the sampled sequence.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import torch
import xarray as xr
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.utils.data import Dataset

from mlcast.data.normalization import NORMALIZATION_REGISTRY


def _time_range_to_index_slice(
    zarr_path: str,
    time_range: tuple[str, str],
    storage_options: dict[str, Any] | None = None,
) -> slice:
    """Convert an inclusive ISO time range into a zarr integer slice.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    time_range : tuple of str
        Inclusive ``(start, end)`` ISO 8601 time range.
    storage_options : dict or None, optional
        Options forwarded to ``xr.open_zarr``. Default is ``None``.

    Returns
    -------
    slice
        Integer slice covering the requested time range.
    """
    ds = xr.open_zarr(zarr_path, storage_options=storage_options)
    time_values = ds.indexes["time"]
    t_start = time_values.get_indexer([pd.Timestamp(time_range[0])], method="bfill")[0]
    t_end = time_values.get_indexer([pd.Timestamp(time_range[1])], method="ffill")[0]
    if t_start < 0 or t_end < 0:
        raise ValueError(
            f"time_range {time_range!r} falls entirely outside the zarr time coordinate "
            f"({time_values[0]} - {time_values[-1]})."
        )
    return slice(int(t_start), int(t_end) + 1)


def _detect_axes(ds: xr.Dataset, standard_name: str) -> tuple[str, str, str]:
    """Detect CF axis dimension names for a variable in an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Open xarray Dataset with CF metadata.
    standard_name : str
        CF standard name of the variable used to infer axes.

    Returns
    -------
    tuple of str
        Names of the time, Y, and X dimensions.
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


class SourceDataSequenceDatasetBase(Dataset, ABC):
    """Abstract base class for source-data-backed sequence datasets.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    standard_names : list of str
        List of CF standard names of variables to load.
    sequence_steps : int
        Number of timesteps to include in each sampled sequence.
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
        sequence_steps: int,
        deterministic: bool = False,
        augment: bool = False,
        width: int = 256,
        height: int = 256,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        if sequence_steps < 1:
            raise ValueError(f"sequence_steps ({sequence_steps}) must be at least 1.")

        self.storage_options = storage_options
        self._zarr_path = zarr_path
        self._ds: xr.Dataset | None = None
        self.standard_names = standard_names
        self.sequence_steps = sequence_steps
        self.augment = augment
        self.w = width
        self.h = height
        self.rng = np.random.default_rng(seed=42) if deterministic else np.random.default_rng(int(time.time()))

        self._validate_standard_names()
        self.t_dim, self.y_dim, self.x_dim = _detect_axes(self.ds, self.standard_names[0])

    @property
    def ds(self) -> xr.Dataset:
        """Open and cache the Zarr-backed xarray Dataset for this worker.

        Returns
        -------
        xr.Dataset
            Opened dataset, optionally subset in time for this worker process.
        """
        if self._ds is None:
            ds = xr.open_zarr(self._zarr_path, storage_options=self.storage_options)
            if self._time_index_slice is not None:
                ds = ds.isel(time=self._time_index_slice)
            self._ds = ds
        return self._ds

    def _validate_standard_names(self) -> None:
        """Check that every requested CF standard name exists in the Zarr store.

        Raises
        ------
        ValueError
            If any requested standard name is missing from the dataset.
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
        self,
        tensor: Float[Tensor, "sequence_steps channels height width"],
        rotate_prob: float = 0.5,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ) -> Float[Tensor, "sequence_steps channels height width"]:
        """Apply random spatial augmentations to a sequence tensor.

        Parameters
        ----------
        tensor : Float[Tensor, "sequence_steps channels height width"]
            Sequence tensor to augment.
        rotate_prob : float, optional
            Probability of applying a random 90-degree rotation. Default is
            ``0.5``.
        hflip_prob : float, optional
            Probability of applying a horizontal flip. Default is ``0.5``.
        vflip_prob : float, optional
            Probability of applying a vertical flip. Default is ``0.5``.

        Returns
        -------
        Float[Tensor, "sequence_steps channels height width"]
            Augmented contiguous tensor.
        """
        if self.rng.random() < rotate_prob:
            k = self.rng.integers(1, 4)
            tensor = torch.rot90(tensor, int(k), dims=[-2, -1])

        if self.rng.random() < hflip_prob:
            tensor = torch.flip(tensor, dims=[-1])

        if self.rng.random() < vflip_prob:
            tensor = torch.flip(tensor, dims=[-2])

        return tensor.contiguous()

    def _build_sequence(self, data: np.ndarray) -> Float[Tensor, "sequence_steps channels height width"]:
        """Convert a raw ``(T, C, H, W)`` numpy array into a tensor.

        Parameters
        ----------
        data : np.ndarray
            Normalized array with shape ``(sequence_steps, channels, height,
            width)``.

        Returns
        -------
        Float[Tensor, "sequence_steps channels height width"]
            Float32 sequence tensor, augmented if requested.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        sequence_t = torch.from_numpy(data)
        if self.augment:
            sequence_t = self._apply_augmentations(sequence_t)
        return sequence_t

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Float[Tensor, "sequence_steps channels height width"]: ...


class SourceDataPrecomputedSequenceDataset(SourceDataSequenceDatasetBase):
    """Sequence dataset using pre-sampled spatial-temporal coordinates from CSV.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    csv_path : str
        Path to the CSV file with ``t``, ``x``, and ``y`` crop coordinates.
    standard_names : list of str
        CF standard names of variables to load.
    sequence_steps : int
        Number of timesteps to include in each sampled sequence.
    deterministic : bool, optional
        If ``True``, use deterministic random sampling within precomputed time
        windows. Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations. Default is ``False``.
    subset : dict or None, optional
        Coordinate subsetting specification. Only ``{"time": (start, end)}``
        is supported. Default is ``None``.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    time_depth : int, optional
        Number of timesteps in each precomputed sampled window. Default is
        ``24``.
    storage_options : dict or None, optional
        Options forwarded to ``xr.open_zarr``. Default is ``None``.
    """

    def __init__(
        self,
        zarr_path: str,
        csv_path: str,
        standard_names: list[str],
        sequence_steps: int,
        deterministic: bool = False,
        augment: bool = False,
        subset: dict[str, Any] | None = None,
        width: int = 256,
        height: int = 256,
        time_depth: int = 24,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        if subset:
            for key in subset:
                if key != "time":
                    raise NotImplementedError(
                        f"subset key {key!r} is not supported. Only 'time' subsetting is currently implemented."
                    )
        time_range: tuple[str, str] | None = (subset or {}).get("time")
        if time_range is not None:
            self._time_index_slice: slice | None = _time_range_to_index_slice(zarr_path, time_range, storage_options)
        else:
            self._time_index_slice = None
        super().__init__(
            zarr_path=zarr_path,
            standard_names=standard_names,
            sequence_steps=sequence_steps,
            deterministic=deterministic,
            augment=augment,
            width=width,
            height=height,
            storage_options=storage_options,
        )

        self.coords = pd.read_csv(csv_path).sort_values("t")
        if self._time_index_slice is not None:
            t_start = self._time_index_slice.start
            t_stop = self._time_index_slice.stop
            self.coords = self.coords[(self.coords["t"] >= t_start) & (self.coords["t"] < t_stop)].reset_index(
                drop=True
            )

        self.dt = time_depth

        if self.sequence_steps > self.dt:
            print(f"Warning: requested sequence_steps ({self.sequence_steps}) > sampled time window ({self.dt})")

        self._ds = None

    def __len__(self) -> int:
        """Get the number of precomputed crop coordinates.

        Returns
        -------
        int
            Number of available sequence samples.
        """
        return len(self.coords)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> Float[Tensor, "sequence_steps channels height width"]:
        """Load and return a single normalized sequence tensor.

        Parameters
        ----------
        idx : int
            Index of the precomputed crop coordinate.

        Returns
        -------
        Float[Tensor, "sequence_steps channels height width"]
            Normalized sequence tensor sampled from the source dataset.
        """
        t0, x0, y0 = self.coords.iloc[idx]

        x_slice = slice(int(x0), int(x0) + self.w)
        y_slice = slice(int(y0), int(y0) + self.h)

        if self.sequence_steps < self.dt:
            t_start = self.rng.integers(t0, t0 + self.dt - self.sequence_steps + 1)
        else:
            t_start = t0
        t_slice = slice(int(t_start), int(t_start) + self.sequence_steps)

        channels = []
        for std_name in self.standard_names:
            da_var = self.ds.cf[std_name].isel({self.t_dim: t_slice, self.x_dim: x_slice, self.y_dim: y_slice})
            norm_func = NORMALIZATION_REGISTRY[std_name]
            channels.append(norm_func(da_var.values))

        data = np.swapaxes(np.stack(channels, axis=0), 0, 1)
        return self._build_sequence(data)


class SourceDataRandomSequenceDataset(SourceDataSequenceDatasetBase):
    """Sequence dataset with on-the-fly random spatial and temporal sampling.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    standard_names : list of str
        CF standard names of variables to load.
    sequence_steps : int
        Number of timesteps to include in each sampled sequence.
    deterministic : bool, optional
        If ``True``, use deterministic random sampling. Default is ``False``.
    augment : bool, optional
        If ``True``, apply random spatial augmentations. Default is ``False``.
    subset : dict or None, optional
        Coordinate subsetting specification. Only ``{"time": (start, end)}``
        is supported. Default is ``None``.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    epoch_size : int, optional
        Number of random samples exposed per epoch. Default is ``1000``.
    storage_options : dict or None, optional
        Options forwarded to ``xr.open_zarr``. Default is ``None``.
    **kwargs : Any
        Ignored extra arguments to allow partial config reuse.
    """

    def __init__(
        self,
        zarr_path: str,
        standard_names: list[str],
        sequence_steps: int,
        deterministic: bool = False,
        augment: bool = False,
        subset: dict[str, Any] | None = None,
        width: int = 256,
        height: int = 256,
        epoch_size: int = 1000,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if subset:
            for key in subset:
                if key != "time":
                    raise NotImplementedError(
                        f"subset key {key!r} is not supported. Only 'time' subsetting is currently implemented."
                    )
        time_range: tuple[str, str] | None = (subset or {}).get("time")
        if time_range is not None:
            self._time_index_slice: slice | None = _time_range_to_index_slice(zarr_path, time_range, storage_options)
        else:
            self._time_index_slice = None
        super().__init__(
            zarr_path=zarr_path,
            standard_names=standard_names,
            sequence_steps=sequence_steps,
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

        if self.sequence_steps > self.max_t:
            raise ValueError(
                f"Requested sequence_steps ({self.sequence_steps}) > available time dimension ({self.max_t})"
            )
        if self.h > self.max_y:
            raise ValueError(f"Requested height ({self.h}) > available Y dimension ({self.max_y})")
        if self.w > self.max_x:
            raise ValueError(f"Requested width ({self.w}) > available X dimension ({self.max_x})")

        self._ds = None

    def __len__(self) -> int:
        """Get the configured random epoch size.

        Returns
        -------
        int
            Number of random sequence samples exposed per epoch.
        """
        return self.epoch_size

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, idx: int) -> Float[Tensor, "sequence_steps channels height width"]:
        """Load and return a single randomly sampled normalized sequence.

        Parameters
        ----------
        idx : int
            Ignored sample index; each call draws a random crop.

        Returns
        -------
        Float[Tensor, "sequence_steps channels height width"]
            Normalized sequence tensor sampled from the source dataset.
        """
        t_start = self.rng.integers(0, self.max_t - self.sequence_steps + 1)
        y_start = self.rng.integers(0, self.max_y - self.h + 1)
        x_start = self.rng.integers(0, self.max_x - self.w + 1)

        t_slice = slice(int(t_start), int(t_start) + self.sequence_steps)
        y_slice = slice(int(y_start), int(y_start) + self.h)
        x_slice = slice(int(x_start), int(x_start) + self.w)

        channels = []
        for std_name in self.standard_names:
            da_var = self.ds.cf[std_name].isel({self.t_dim: t_slice, self.x_dim: x_slice, self.y_dim: y_slice})
            norm_func = NORMALIZATION_REGISTRY[std_name]
            channels.append(norm_func(da_var.values))

        data = np.swapaxes(np.stack(channels, axis=0), 0, 1)
        return self._build_sequence(data)

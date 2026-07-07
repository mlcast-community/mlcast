"""PyTorch datasets for loading spatio-temporal data from Zarr stores.

Provides an indexed dataset (crops from a precomputed index) and a
random-sampling dataset.
"""

import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import xarray as xr
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch.utils.data import Dataset

from mlcast.data.normalization import NORMALIZATION_REGISTRY
from mlcast.sampling import Sampler


def _load_sampling_index(path: str) -> pd.DataFrame:
    """Load a precomputed sampling index as a DataFrame.

    Accepts a stats parquet (the dataset sampler's output) or a legacy
    ``.csv``. Returns at least the ``t, x, y`` crop-corner columns, plus the
    per-datacube ``mean`` column when the file carries it (parquet only) — the
    latter feeds importance sampling. Only the needed columns are read.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        available = set(pq.read_schema(path).names)
        columns = [c for c in ("t", "x", "y", "mean") if c in available]
        return pq.read_table(path, columns=columns).to_pandas()
    raise ValueError(f"Unsupported sampling index format {suffix!r} for {path!r}; expected .parquet or .csv")


def _time_range_to_index_slice(
    zarr_path: str,
    time_range: tuple[str, str],
    storage_options: dict[str, Any] | None = None,
) -> slice:
    """Convert an inclusive ISO time range into a zarr integer slice."""
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


class DatasetSample(TypedDict, total=False):
    """Typed dictionary returned by dataset ``__getitem__``.

    Keys
    ----
    input : Float[torch.Tensor, "input_steps channels height width"]
        Past frames fed to the network as input.
    target : Float[torch.Tensor, "forecast_steps channels height width"]
        Future frames the network should predict.
    target_mask : Float[torch.Tensor, "1 channels height width"]
        Per-cell validity mask collapsed over the whole sequence: a cell is 1
        only if it was finite at every step (inputs and targets), else 0. Its
        leading axis is a single step that the loss broadcasts over the forecast
        steps. Only present when ``return_mask=True``.
    """

    input: Float[torch.Tensor, "input_steps channels height width"]
    target: Float[torch.Tensor, "forecast_steps channels height width"]
    target_mask: Float[torch.Tensor, "1 channels height width"]


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
            if self._time_index_slice is not None:
                ds = ds.isel(time=self._time_index_slice)
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
        # Validity mask, collapsed over the whole sequence: a cell is scored only
        # if it is finite at EVERY step (inputs and targets). A temporal
        # discontinuity anywhere makes the forecast trajectory at that cell
        # ill-defined — and the temporal-consistency loss term meaningless — so we
        # mask it for the whole sequence. Kept as (1, C, H, W); the loss broadcasts
        # it over the forecast steps, so no T copies are materialised on the GPU.
        # Computed before NaNs are filled below.
        if self.return_mask:
            valid = ~np.isnan(data).any(axis=0, keepdims=True)  # (1, C, H, W)
            target_mask_t = torch.from_numpy(valid.astype(np.float32))

        # source data may be float64, but the model and the rest of the
        # training pipeline operate in float32.
        data = np.nan_to_num(data, nan=-1.0).astype(np.float32)
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


class SourceDataIndexedDataset(SourceDataDatasetBase):
    """PyTorch dataset yielding Zarr crops at locations read from a precomputed index.

    Each sample is a spatio-temporal crop of shape ``(T, C, H, W)``
    converted to normalized data.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    index_path : str
        Path to the sampling index of ``(t, x, y)`` crop corners: a stats
        parquet (the candidate pool, optionally filtered by ``sampler``) or a
        legacy ``.csv`` (already sampled, used as-is).
    standard_names : list of str
        List of CF standard names of variables to load (e.g., ``["rainfall_flux"]``).
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
    subset : dict or None, optional
        Coordinate subsetting specification. Only ``{"time": (start, end)}``
        is supported, where the time range is inclusive and uses ISO strings.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    time_depth : int, optional
        Number of timesteps in the sampled window. Default is ``24``.
    sampler : Sampler or None, optional
        Optional sampler to filter the candidate index with a chosen strategy,
        applied once at init (see :mod:`mlcast.sampling.samplers`). Default
        ``None`` keeps every candidate.
    sampling_seed : int, optional
        Seed for the sampler's one-time selection. Default ``42``.
    """

    def __init__(
        self,
        zarr_path: str,
        index_path: str,
        standard_names: list[str],
        input_steps: int,
        forecast_steps: int,
        return_mask: bool = False,
        deterministic: bool = False,
        augment: bool = False,
        subset: dict[str, Any] | None = None,
        width: int = 256,
        height: int = 256,
        time_depth: int = 24,
        storage_options: dict[str, Any] | None = None,
        sampler: Sampler | None = None,
        sampling_seed: int = 42,
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
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            return_mask=return_mask,
            deterministic=deterministic,
            augment=augment,
            width=width,
            height=height,
            storage_options=storage_options,
        )

        self.coords = _load_sampling_index(index_path).sort_values("t")
        if self._time_index_slice is not None:
            t_start = self._time_index_slice.start
            t_stop = self._time_index_slice.stop
            # `t` is an absolute index into the full zarr, but `self.ds` is sliced to
            # this subset (its time axis is 0-based). Keep only windows whose full
            # depth fits inside the subset, then rebase `t` onto the sliced axis so
            # `__getitem__` indexes it correctly and splits don't leak across the
            # boundary.
            self.coords = self.coords[
                (self.coords["t"] >= t_start) & (self.coords["t"] + time_depth <= t_stop)
            ].reset_index(drop=True)
            self.coords["t"] = self.coords["t"] - t_start

        if sampler is not None:
            selected = sampler.select(self.coords, np.random.default_rng(sampling_seed))
            self.coords = self.coords.iloc[selected].reset_index(drop=True)

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
        row = self.coords.iloc[idx]
        t0, x0, y0 = row["t"], row["x"], row["y"]

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

        # swapaxes returns a view; make it contiguous and float32 before
        # handing it to _build_sample()/torch.from_numpy().
        data = np.ascontiguousarray(np.swapaxes(np.stack(channels, axis=0), 0, 1), dtype=np.float32)
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
        List of CF standard names of variables to load (e.g., ``["rainfall_flux"]``).
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
    subset : dict or None, optional
        Coordinate subsetting specification. Only ``{"time": (start, end)}``
        is supported, where the time range is inclusive and uses ISO strings.
    width : int, optional
        Spatial width of each crop. Default is ``256``.
    height : int, optional
        Spatial height of each crop. Default is ``256``.
    epoch_size : int, optional
        Number of random samples to generate per epoch. Default is ``1000``.
    **kwargs : Any
        Ignored extra arguments (e.g. ``index_path``) to allow drop-in replacement.
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

        # swapaxes returns a view; make it contiguous and float32 before
        # handing it to _build_sample()/torch.from_numpy().
        data = np.ascontiguousarray(np.swapaxes(np.stack(channels, axis=0), 0, 1), dtype=np.float32)
        return self._build_sample(data)

"""Per-datacube statistics via cumsum-based sliding windows.

Scans a Zarr dataset for valid datacube candidates and computes, for each
one, `nan_count`, `sum`, `mean`, and `frac_wet`, each in O(1) per window
amortized via a prefix-sum (cumsum) trick. The survivors (those passing
the `max_nan`, stride, and time-continuity filters) are written to a stats
parquet whose contract is defined in `stats_spec`. Downstream,
`SourceDataIndexedDataset` reads this parquet and an `ImportanceSampler`
(see `samplers`) selects candidates by the `mean` column; no separate
sampling pass is needed.

Heavy stats that cannot be computed with cumsum (max, quantiles) are out
of scope here and could be added as extra columns in a future pass.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING

import bottleneck as bn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
from loguru import logger
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..console import console
from ..stats_spec import STAT_COLUMNS, StatsMetadata, build_schema
from ..units import default_wet_threshold, detect_data_kind

if TYPE_CHECKING:
    from numpy.typing import NDArray


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add stats specific arguments to the parser."""
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr dataset.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output Parquet file path. If not specified, auto-generated from parameters.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--time-depth", type=int, default=24, help="Time depth of datacubes.")
    parser.add_argument("--width", type=int, default=256, help="Spatial width of datacubes.")
    parser.add_argument("--height", type=int, default=256, help="Spatial height of datacubes.")
    parser.add_argument("--step-t", type=int, default=3, help="Time step between datacubes.")
    parser.add_argument("--step-x", type=int, default=16, help="X step between datacubes.")
    parser.add_argument("--step-y", type=int, default=16, help="Y step between datacubes.")
    parser.add_argument(
        "--max-nan",
        type=int,
        default=10000,
        help="Maximum NaN count per datacube (hard filter on output).",
    )
    parser.add_argument(
        "--wet-threshold",
        type=float,
        default=None,
        help="Wet-pixel threshold in the same units as the data var. "
        "If omitted, auto-detected: 0.1 mm/h for rain rate, 7 dBZ for reflectivity.",
    )
    parser.add_argument(
        "--data-kind",
        choices=["rainrate", "reflectivity"],
        default=None,
        help="Override the data-kind auto-detection from zarr attrs. "
        "Needed only if the variable has non-standard attributes.",
    )
    parser.add_argument(
        "--time-step-minutes",
        type=int,
        default=5,
        help="Expected time step between consecutive frames in minutes.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute backend. 'auto' (default) uses CUDA if a GPU is available, else the CPU. "
        "'cuda' requires a CUDA GPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU: number of worker processes. GPU: number of chunk-reader threads.",
    )
    parser.add_argument("--data-var", type=str, default="RR", help="Name of the zarr data variable.")
    parser.add_argument("--time-var", type=str, default="time", help="Name of the zarr time variable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")


def _dim_cumsum_window(
    arr: NDArray,
    dim: int,
    delta: int,
    dim_len: int,
) -> NDArray:
    """3D sliding-window sum along one axis via a prefix-sum difference.

    Works for any numeric dtype (int for counting, float for summing). For
    every window of size `delta` along `dim`, returns the sum of the
    elements inside that window. O(n) per axis regardless of `delta`.

    The output length along `dim` is ``dim_len - delta + 1`` (one entry per
    valid window start, including the final one at ``dim_len - delta``).
    """
    # Use int32 (not the numpy default int64) for int inputs to halve memory.
    # A window count is bounded by Dt*w*h <= 2^31, safe in int32.
    cumsum = np.cumsum(arr, axis=dim, dtype=arr.dtype if arr.dtype.kind == "f" else np.int32)

    # Prepend a zero plane so window [k, k+delta) == padded[k+delta] - padded[k].
    pad_width = [(1, 0) if i == dim else (0, 0) for i in range(arr.ndim)]
    padded = np.pad(cumsum, pad_width=pad_width, mode="constant", constant_values=0)

    # Window starts k = 0 .. dim_len-delta, so padded[k] spans [0, dim_len-delta]
    # and padded[k+delta] spans [delta, dim_len] (note the inclusive +1 ends).
    slices_start = [slice(dim_len - delta + 1) if i == dim else slice(None) for i in range(arr.ndim)]
    slices_end = [slice(delta, dim_len + 1) if i == dim else slice(None) for i in range(arr.ndim)]

    return padded[tuple(slices_end)] - padded[tuple(slices_start)]


def _datacube_window_sum(
    arr: NDArray,
    deltas: tuple[int, int, int],
    dim_lengths: tuple[int, int, int],
    order: tuple[int, int, int] = (0, 1, 2),
) -> NDArray:
    """3-axis cumsum-window sum over a (T, X, Y) array (all positions).

    `order` is the sequence in which the axes are windowed. Windowing one
    axis only shrinks that axis, so for an associative reduction the result
    is independent of `order`. The production path uses `_strided_window`
    instead (it evaluates only the kept positions); this full-cube variant
    is kept for tests and reference.
    """
    s = arr
    for ax in order:
        s = _dim_cumsum_window(s, dim=ax, delta=deltas[ax], dim_len=dim_lengths[ax])
    return s


def _strided_window(
    arr: NDArray,
    deltas: tuple[int, int, int],
    dim_lengths: tuple[int, int, int],
    off_t: int,
    steps: tuple[int, int, int],
    keep_t: NDArray[np.bool_],
) -> NDArray:
    """Windowed sum reduced to the strided, gap-free candidate grid.

    Windows the time axis, keeps only the strided, gap-free t-slices, then
    windows x and y on the already-strided (smaller) arrays. Selecting the
    candidate grid *between* the axes shrinks the x cumsum by ``step_t`` and
    the y cumsum by ``step_t * step_x`` while leaving the kept values
    unchanged: windowing one axis is independent of which positions are kept
    on the others.

    The full-array time axis uses bottleneck's ``move_sum`` (faster than a
    numpy cumsum on this large, non-contiguous axis); the small strided
    spatial axes stay on numpy. ``move_sum`` needs float input and emits NaN
    for the first ``Dt-1`` incomplete windows, dropped with ``[Dt-1:]``;
    integer counts are cast back to int32 (exact to 2^31).

    `keep_t` is the boolean continuity mask over the strided t-grid
    ``arange(off_t, dim_lengths[0] - deltas[0] + 1, step_t)``; its length
    must match that grid.
    """
    Dt, w, h = deltas
    step_t, step_x, step_y = steps
    s = bn.move_sum(arr.astype(np.float32, copy=False), Dt, axis=0)[Dt - 1 :]
    if arr.dtype.kind in "bi":
        s = s.astype(np.int32)
    s = s[off_t::step_t][keep_t]
    s = _dim_cumsum_window(s, dim=1, delta=w, dim_len=dim_lengths[1])
    s = s[:, 0::step_x]
    s = _dim_cumsum_window(s, dim=2, delta=h, dim_len=dim_lengths[2])
    return s[:, :, 0::step_y]


def _dominant_time_step(time_diffs: pd.TimedeltaIndex) -> tuple[pd.Timedelta, float]:
    """Most frequent step between consecutive frames and its share of all steps."""
    values, counts = np.unique(time_diffs, return_counts=True)
    i = int(np.argmax(counts))
    return pd.Timedelta(values[i]), counts[i] / len(time_diffs)


def _process_chunk(
    time_range: tuple[int, int],
    t_start_idx: int,
    data: zarr.Array,
    max_nan: int,
    wet_threshold: float,
    deltas: tuple[int, int, int],
    steps: tuple[int, int, int],
    valid_start_mask: NDArray[np.bool_],
) -> dict[str, NDArray]:
    """Compute the windowed stats for one time chunk's candidate datacubes.

    Returns the survivors — windows passing the `max_nan`, stride, and
    time-continuity filters — with their `nan_count`, `sum`, `mean`, and
    `frac_wet`, as a dict of numpy arrays matching `STATS_SCHEMA`.

    The three reductions (nan_count, sum, wet_count) are each computed with
    `_strided_window` and freed before the next, keeping peak memory near a
    single windowed array. `valid_start_mask` is a boolean lookup over the
    filtered time axis (True at gap-free window starts).
    """
    start_t, end_t = time_range
    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :].astype(np.float32, copy=False)
    dim_lengths = chunk.shape
    # Data axes follow the MLCast source-data spec (§4.3): (time, y=height, x=width).
    Dt, dy, dx = deltas
    step_t, step_y, step_x = steps
    total_px = Dt * dy * dx

    # Strided, gap-free time-window starts, absolute-index aligned. The time
    # window count is dim_lengths[0] - Dt + 1 (see _dim_cumsum_window); the
    # x/y stride offsets are 0 because chunk x/y are already absolute.
    off_t = (-(start_t + t_start_idx)) % step_t
    t_rel_strided = np.arange(off_t, dim_lengths[0] - Dt + 1, step_t, dtype=np.int32)
    keep_t = valid_start_mask[t_rel_strided + start_t]  # time-continuity filter
    t_rel_kept = t_rel_strided[keep_t]

    # Build the NaN mask once; (a) drives nan_count, (b) zero-fills the sum pass.
    nan_mask = np.isnan(chunk)  # bool, 1 byte/element

    # --- Pass A: nan_count on the strided candidate grid -------------------------
    ncw_s = _strided_window(nan_mask, deltas, dim_lengths, off_t, steps, keep_t)
    surv_t, surv_y, surv_x = np.where(ncw_s <= max_nan)
    nan_count = ncw_s[surv_t, surv_y, surv_x].astype(np.int32)
    del ncw_s

    # Map strided survivor indices back to chunk-relative / absolute coords.
    idx_t_rel = t_rel_kept[surv_t]
    idx_y = (surv_y * step_y).astype(np.int32)
    idx_x = (surv_x * step_x).astype(np.int32)
    idx_t_abs = (idx_t_rel + (start_t + t_start_idx)).astype(np.int32)

    # --- Pass B: sum -------------------------------------------------------------
    chunk[nan_mask] = 0.0
    sum_vals = _strided_window(chunk, deltas, dim_lengths, off_t, steps, keep_t)[surv_t, surv_y, surv_x]

    # --- Pass C: wet_count -------------------------------------------------------
    # `chunk` is now zero where it was NaN, so `> wet_threshold` is equivalent
    # to (value > threshold AND not NaN).
    wet_mask = chunk > wet_threshold
    del chunk, nan_mask
    wet_count = _strided_window(wet_mask, deltas, dim_lengths, off_t, steps, keep_t)[surv_t, surv_y, surv_x]

    # Derived stats.
    valid_count = total_px - nan_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vals = np.where(valid_count > 0, sum_vals / valid_count, np.nan).astype(np.float32)
    frac_wet = wet_count.astype(np.float32) / total_px

    return {
        "t": idx_t_abs,
        "x": idx_x,
        "y": idx_y,
        "nan_count": nan_count,
        "sum": sum_vals.astype(np.float32),
        "mean": mean_vals,
        "frac_wet": frac_wet,
    }


def _parquet_writer(
    output_queue: Queue,
    filename: str,
    schema: pa.Schema,
) -> None:
    """Drain the queue and stream rows to a Parquet file.

    Each queue item is the dict returned by `_process_chunk`. We buffer
    into Arrow RecordBatches and append to a single ParquetWriter so the
    on-disk file stays a single self-contained parquet. `schema` is the
    canonical STATS_SCHEMA with the mlcast sampling parameters attached as
    metadata (see `stats_spec.build_schema`), so downstream commands don't
    need to parse the filename.
    """
    # Column encodings tuned to the data: `t` is written in ascending order so
    # it delta-encodes to almost nothing; x/y/nan_count are low-cardinality
    # (dictionary); the floats compress better split into byte planes.
    writer = pq.ParquetWriter(
        filename,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=["x", "y", "nan_count"],
        column_encoding={
            "t": "DELTA_BINARY_PACKED",
            "sum": "BYTE_STREAM_SPLIT",
            "mean": "BYTE_STREAM_SPLIT",
            "frac_wet": "BYTE_STREAM_SPLIT",
        },
    )
    total_rows = 0
    try:
        while True:
            item = output_queue.get()
            if item is None:
                break
            if item["t"].size == 0:
                continue
            batch = pa.record_batch(
                [pa.array(item[c]) for c in STAT_COLUMNS],
                schema=schema,
            )
            writer.write_batch(batch)
            total_rows += batch.num_rows
    finally:
        writer.close()
    logger.info(f"Wrote {total_rows} rows to {filename}")


def _resolve_device(requested: str) -> tuple[str, str]:
    """Resolve the compute backend to ('cpu'|'cuda', human label).

    'auto' picks CUDA when PyTorch + a GPU are importable/available, else CPU.
    'cuda' raises ValueError if PyTorch or a GPU is missing. 'cpu' is forced.
    """
    if requested == "cpu":
        return "cpu", "cpu (bottleneck)"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise ValueError("--device cuda requested but PyTorch is not installed.") from None
        return "cpu", "cpu (bottleneck)"
    if torch.cuda.is_available():
        return "cuda", f"cuda ({torch.cuda.get_device_name(0)})"
    if requested == "cuda":
        raise ValueError("--device cuda requested but no CUDA GPU is available.")
    return "cpu", "cpu (bottleneck)"


def _prefetched(read_fn, items, lookahead: int, n_workers: int):
    """Yield ``read_fn(item)`` results in submission order, keeping up to
    `lookahead` reads in flight across `n_workers` threads.

    Used by the GPU path to overlap chunk reads/decompression (which release
    the GIL) with GPU compute, so the device stays fed. Memory is bounded by
    `lookahead` chunks.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        it = iter(items)
        inflight: deque = deque()
        for _ in range(lookahead):
            try:
                inflight.append(ex.submit(read_fn, next(it)))
            except StopIteration:
                break
        while inflight:
            fut = inflight.popleft()
            try:
                inflight.append(ex.submit(read_fn, next(it)))
            except StopIteration:
                pass
            yield fut.result()


def run(args: argparse.Namespace) -> int:
    """Execute the stats command."""
    start_time = time.time()
    Dt = args.time_depth
    w = args.width
    h = args.height
    step_T = args.step_t
    step_X = args.step_x
    step_Y = args.step_y
    max_nan = args.max_nan
    n_workers = args.workers
    time_chunk_size = 3 * Dt

    # Window extents and strides in the array's (time, y, x) axis order. Per the
    # MLCast source-data spec the spatial axes are y=height (axis 1) and
    # x=width (axis 2), so height/step_y bind to axis 1 and width/step_x to axis 2.
    deltas = (Dt, h, w)
    steps = (step_T, step_Y, step_X)

    try:
        device, device_label = _resolve_device(args.device)
    except ValueError as e:
        logger.error(str(e))
        return 1
    logger.info(f"Compute backend: {device_label}")

    logger.info(f"Opening Zarr dataset: {args.zarr_path}")
    try:
        zg = zarr.open(args.zarr_path, mode="r")
        data = zg[args.data_var]
        ds = xr.open_zarr(args.zarr_path)
        time_array_full = pd.DatetimeIndex(ds[args.time_var].values)
        logger.info(f"Full dataset shape: T={data.shape[0]}, Y={data.shape[1]}, X={data.shape[2]}")
        logger.info(f"Time range: {time_array_full[0]} to {time_array_full[-1]}")
        var_attrs = dict(ds[args.data_var].attrs)
    except Exception as e:
        logger.error(f"Error loading Zarr dataset: {e}")
        return 1

    # Detect or override the data kind, then resolve the wet-pixel threshold.
    if args.data_kind is not None:
        data_kind = args.data_kind
        logger.info(f"Data kind overridden via --data-kind: {data_kind}")
    else:
        try:
            data_kind = detect_data_kind(var_attrs)
        except ValueError as e:
            logger.error(str(e))
            return 1
        logger.info(
            f"Detected data kind: {data_kind} "
            f"(standard_name={var_attrs.get('standard_name')!r}, "
            f"units={var_attrs.get('units')!r})"
        )

    wet_threshold = args.wet_threshold if args.wet_threshold is not None else default_wet_threshold(data_kind)
    units_str = var_attrs.get("units", "?")
    logger.info(f"Wet-pixel threshold: {wet_threshold} {units_str}")

    start_date = pd.to_datetime(args.start_date) if args.start_date else time_array_full[0]
    end_date = pd.to_datetime(args.end_date) if args.end_date else time_array_full[-1]

    mask = (time_array_full >= start_date) & (time_array_full <= end_date)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        logger.error(f"No data found between {start_date} and {end_date}")
        return 1

    t_start_idx = valid_indices[0]
    t_end_idx = valid_indices[-1] + 1
    size_T = t_end_idx - t_start_idx
    size_Y = data.shape[1]  # axis 1 = y (height), per the source-data spec
    size_X = data.shape[2]  # axis 2 = x (width)
    time_array = time_array_full[t_start_idx:t_end_idx]

    logger.info(f"Filtered dataset shape: T={size_T}, X={size_X}, Y={size_Y}")
    logger.info(f"Filtered time range: {time_array[0]} to {time_array[-1]}")
    max_t = size_T - Dt + 1
    if max_t < 1:
        logger.error(
            f"The selected time range holds only {size_T} frames but --time-depth is {Dt}: "
            "no datacube fits. Widen the date range or reduce --time-depth."
        )
        return 1

    logger.info("Checking time continuity...")
    expected_step = pd.Timedelta(minutes=args.time_step_minutes)
    time_diffs = time_array[1:] - time_array[:-1]
    gaps = (time_diffs != expected_step).astype(int)
    window_sum = np.convolve(gaps, np.ones(Dt - 1, dtype=int), mode="valid")
    valid_starts_gap = np.where(window_sum == 0)[0]
    logger.info(f"Found {len(valid_starts_gap)} valid time starts without gaps")
    if len(valid_starts_gap) == 0:
        dominant, share = _dominant_time_step(time_diffs)
        if dominant != expected_step:
            dominant_min = dominant / pd.Timedelta(minutes=1)
            hint = (
                f"If that is the dataset's real cadence, rerun with --time-step-minutes {dominant_min:g}."
                if dominant_min == int(dominant_min)
                else "Set --time-step-minutes to the dataset's real cadence."
            )
            logger.error(
                f"0 gap-free time windows: expecting a {args.time_step_minutes}-min step between "
                f"frames (--time-step-minutes) but the dominant observed step is {dominant_min:g} min "
                f"({share:.1%} of {len(time_diffs):,} steps). {hint}"
            )
        else:
            logger.error(
                f"0 gap-free time windows: the time axis matches the expected "
                f"{args.time_step_minutes}-min step, but every window of {Dt} frames spans a gap. "
                "Reduce --time-depth or pick a date range with fewer missing frames."
            )
        return 1
    # Boolean lookup over the filtered time axis for an O(1) continuity test
    # per candidate window start.
    valid_start_mask = np.zeros(size_T, dtype=bool)
    valid_start_mask[valid_starts_gap] = True

    # Peak memory per worker ~= chunk (float32) + cumsum working set + nan_mask (bool).
    # The three cumsum reductions run sequentially, so only one window array is
    # alive at a time.
    chunk_bytes = (time_chunk_size + Dt - 1) * size_X * size_Y * 4
    per_chunk_gb = 2 * chunk_bytes / (1024**3)
    logger.info(f"Estimated memory per chunk: {per_chunk_gb:.2f} GB (pipelined cumsums)")
    logger.info(f"Estimated total memory ({n_workers} workers): {per_chunk_gb * n_workers:.2f} GB")

    t_starts = np.arange(0, max_t, time_chunk_size)
    t_ends = np.minimum(t_starts + time_chunk_size + Dt - 1, size_T)
    t_pairs = np.stack((t_starts, t_ends), axis=1)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    if args.output:
        output_file = args.output
    else:
        output_file = f"stats_{start_str}-{end_str}_{Dt}x{w}x{h}_{step_T}x{step_X}x{step_Y}_{max_nan}.parquet"
    if os.path.exists(output_file) and not args.overwrite:
        logger.error(f"File {output_file} already exists. Use --overwrite to replace.")
        return 1
    logger.info(f"Output file: {output_file}")

    metadata = StatsMetadata(
        zarr_path=args.zarr_path,
        data_var=args.data_var,
        time_var=args.time_var,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        time_step_minutes=args.time_step_minutes,
        time_depth=Dt,
        width=w,
        height=h,
        step_t=step_T,
        step_x=step_X,
        step_y=step_Y,
        max_nan=max_nan,
        wet_threshold=wet_threshold,
        data_kind=data_kind,
        units=var_attrs.get("units"),
    )
    schema = build_schema(metadata)

    cfg = Table.grid(padding=(0, 2))
    cfg.add_column(justify="right", style="bold cyan")
    cfg.add_column()
    cfg.add_row("Dataset", f"📦  T={size_T:,}  X={size_X:,}  Y={size_Y:,}")
    cfg.add_row("Time range", f"📅  {time_array[0]:%Y-%m-%d %H:%M} → {time_array[-1]:%Y-%m-%d %H:%M}")
    cfg.add_row("Time step", f"🕒  {args.time_step_minutes} min")
    cfg.add_row("Datacube", f"🧊  {Dt} × {w} × {h}   stride {step_T} × {step_X} × {step_Y}")
    cfg.add_row("Valid starts", f"✅  {len(valid_starts_gap):,} gap-free")
    cfg.add_row("Filters", f"🔍  max_nan={max_nan:,}   wet > {wet_threshold:g} {units_str}")
    cfg.add_row("Data kind", f"💧  {data_kind}")
    cfg.add_row("Device", f"⚡  {device_label}")
    if device == "cuda":
        cfg.add_row("Readers", f"🧵  {n_workers} threads")
    else:
        cfg.add_row("Workers", f"🧵  {n_workers}   ~{per_chunk_gb * n_workers:.1f} GB peak")
    cfg.add_row("Output", f"💾  {output_file}")
    console.print(
        Panel(
            cfg,
            title="[bold]📊 mlcast stats[/]",
            subtitle=f"[dim]{os.path.basename(args.zarr_path)}[/]",
            border_style="blue",
            expand=False,
        )
    )

    output_queue: Queue = Queue(maxsize=100)
    writer_thread = Thread(target=_parquet_writer, args=(output_queue, output_file, schema))
    writer_thread.daemon = False
    writer_thread.start()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    if device == "cuda":
        import torch

        from . import _stats_gpu

        dev = torch.device("cuda")

        def _read_chunk(tp):
            s0, e0 = int(tp[0]), int(tp[1])
            arr = np.asarray(data[s0 + t_start_idx : e0 + t_start_idx, :, :], dtype=np.float32)
            return (s0, e0), arr

        with progress:
            task = progress.add_task("🔍 Scanning time chunks (GPU)", total=len(t_starts))
            for time_range, chunk_np in _prefetched(
                _read_chunk, t_pairs, lookahead=n_workers + 2, n_workers=max(1, n_workers)
            ):
                hits = _stats_gpu.process_chunk(
                    time_range,
                    t_start_idx,
                    chunk_np,
                    max_nan,
                    wet_threshold,
                    deltas,
                    steps,
                    valid_start_mask,
                    dev,
                )
                output_queue.put(hits)
                progress.advance(task)
    else:
        process_chunk_partial = partial(
            _process_chunk,
            t_start_idx=t_start_idx,
            data=data,
            max_nan=max_nan,
            wet_threshold=wet_threshold,
            deltas=deltas,
            steps=steps,
            valid_start_mask=valid_start_mask,
        )
        with progress:
            task = progress.add_task("🔍 Scanning time chunks", total=len(t_starts))
            with Pool(n_workers) as pool:
                for hits in pool.imap(process_chunk_partial, t_pairs, chunksize=1):
                    output_queue.put(hits)
                    progress.advance(task)

    output_queue.put(None)
    writer_thread.join()

    n_rows = pq.read_metadata(output_file).num_rows
    if n_rows == 0:
        console.print(
            Panel(
                f"⚠️  0 datacube candidates survived the filters: every {Dt} × {w} × {h} window "
                f"holds more than max_nan={max_nan:,} NaNs.\n"
                "The window may be larger than the radar coverage, or the selected period all-NaN. "
                "Raise --max-nan or shrink --width/--height/--time-depth.\n"
                f"[dim]💾 {output_file} (empty)[/]",
                title="[bold yellow]⚠️ stats found nothing[/]",
                border_style="yellow",
                expand=False,
            )
        )
        return 1
    console.print(
        Panel(
            f"✅ Wrote [bold]{n_rows:,}[/] datacube candidates "
            f"in [bold]{time.time() - start_time:.1f}s[/]\n"
            f"[dim]💾 {output_file}[/]",
            title="[bold green]🎉 stats complete[/]",
            border_style="green",
            expand=False,
        )
    )
    return 0

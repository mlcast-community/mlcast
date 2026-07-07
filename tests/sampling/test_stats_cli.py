"""Failure-mode tests for the `stats` command's `run()` entry point.

A stats run that can produce no candidates must fail loudly (non-zero exit)
instead of silently writing an empty parquet — the classic trigger being a
dataset whose cadence differs from --time-step-minutes (e.g. 10-minute DMI
data against the 5-minute default), where the continuity filter rejects
every window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr

from mlcast.sampling.commands.stats import add_arguments, run

T_TOTAL = 30


def _write_zarr(path: Path, times: pd.DatetimeIndex, all_nan: bool = False) -> Path:
    rng = np.random.default_rng(0)
    data = rng.gamma(0.5, 2.0, size=(len(times), 40, 40)).astype(np.float32)
    if all_nan:
        data[:] = np.nan
    ds = xr.Dataset(
        {"RR": (("time", "y", "x"), data)},
        coords={"time": times, "y": np.arange(40), "x": np.arange(40)},
    )
    ds["RR"].attrs = {"standard_name": "rainfall_flux", "units": "mm/h"}
    ds.to_zarr(path, mode="w")
    return path


@pytest.fixture
def zarr_10min(tmp_path: Path) -> Path:
    """A tiny, gap-free zarr at 10-minute cadence (DMI-like)."""
    return _write_zarr(tmp_path / "ten_min.zarr", pd.date_range("2024-01-01", periods=T_TOTAL, freq="10min"))


def _run_stats(zarr_path: Path, output: Path, *extra_cli: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(
        [
            str(zarr_path),
            "-o",
            str(output),
            "--device",
            "cpu",
            "--workers",
            "1",
            "--time-depth",
            "4",
            "--width",
            "16",
            "--height",
            "16",
            "--step-t",
            "1",
            "--step-x",
            "8",
            "--step-y",
            "8",
            "--max-nan",
            "100",
            *extra_cli,
        ]
    )
    return run(args)


def test_cadence_mismatch_fails_loudly(zarr_10min: Path, tmp_path: Path) -> None:
    """10-minute data against the 5-minute default: every frame pair looks like
    a gap, so instead of writing an empty parquet the command must error out."""
    out = tmp_path / "stats.parquet"
    assert _run_stats(zarr_10min, out) == 1
    assert not out.exists()


def test_matching_cadence_succeeds(zarr_10min: Path, tmp_path: Path) -> None:
    out = tmp_path / "stats.parquet"
    assert _run_stats(zarr_10min, out, "--time-step-minutes", "10") == 0
    assert pq.read_metadata(out).num_rows > 0


def test_time_range_shorter_than_depth_fails(zarr_10min: Path, tmp_path: Path) -> None:
    out = tmp_path / "stats.parquet"
    code = _run_stats(zarr_10min, out, "--time-step-minutes", "10", "--time-depth", str(T_TOTAL + 1))
    assert code == 1
    assert not out.exists()


def test_gappy_axis_with_matching_cadence_fails(tmp_path: Path) -> None:
    """Cadence matches, but a recurring gap breaks every window of 4 frames."""
    steps_min = np.tile([10, 10, 30], T_TOTAL // 3)[: T_TOTAL - 1]
    offsets_min = np.concatenate([[0], steps_min.cumsum()])
    times = pd.DatetimeIndex(pd.Timestamp("2024-01-01") + pd.to_timedelta(offsets_min, unit="m"))
    store = _write_zarr(tmp_path / "gappy.zarr", times)
    out = tmp_path / "stats.parquet"
    assert _run_stats(store, out, "--time-step-minutes", "10") == 1
    assert not out.exists()


def test_all_windows_filtered_out_fails(tmp_path: Path) -> None:
    """Valid time axis but every window exceeds max_nan: exit non-zero, and the
    (empty) parquet is left on disk for inspection."""
    store = _write_zarr(
        tmp_path / "all_nan.zarr", pd.date_range("2024-01-01", periods=T_TOTAL, freq="10min"), all_nan=True
    )
    out = tmp_path / "stats.parquet"
    code = _run_stats(store, out, "--time-step-minutes", "10", "--max-nan", "0")
    assert code == 1
    assert out.exists()
    assert pq.read_metadata(out).num_rows == 0

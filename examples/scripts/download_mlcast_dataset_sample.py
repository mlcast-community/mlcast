# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray",
#     "zarr",
#     "s3fs",
#     "pandas",
#     "loguru",
#     "mlcast-datasets",
#     "dask"
# ]
# ///
"""Download a temporal subset of a dataset from the mlcast-datasets catalog.

This script uses the official `mlcast_datasets` catalog to discover and download
a specific temporal slice of a remote dataset. It mirrors the exact remote S3
directory structure onto your local filesystem.

Usage:
    uv run examples/scripts/download_mlcast_dataset_sample.py \
        precipitation.radklim_5_minutes \
        --duration PT1H

    If `--start-time` is omitted, the script defaults to the first available
    timestamp in the remote dataset.
"""

import argparse
import sys
from pathlib import Path

import mlcast_datasets
import pandas as pd
from loguru import logger


def get_local_mirrored_path(node, base_dir: str) -> Path:
    """Extract the remote URL from the Intake node and mirror it locally."""
    # Depending on the Intake driver, the urlpath might be hidden in different attributes
    urlpath = None
    if hasattr(node, "urlpath"):
        urlpath = node.urlpath
    elif hasattr(node, "_urlpath"):
        urlpath = node._urlpath
    elif (
        hasattr(node, "_entry")
        and hasattr(node._entry, "describe")
        and "urlpath" in node._entry.describe().get("args", {})
    ):
        urlpath = node._entry.describe()["args"]["urlpath"]
    elif hasattr(node, "describe") and "urlpath" in node.describe().get("args", {}):
        urlpath = node.describe()["args"]["urlpath"]

    if urlpath is None:
        raise ValueError(f"Could not determine the remote URL path for catalog node: {node.name}")

    # Strip scheme (e.g., 's3://', 'https://')
    if "://" in urlpath:
        urlpath = urlpath.split("://", 1)[1]

    # Ensure it doesn't start with a slash so it joins properly
    urlpath = urlpath.lstrip("/")

    return Path(base_dir) / urlpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a temporal slice from the mlcast-datasets catalog.")
    parser.add_argument(
        "item",
        type=str,
        help="The nested item path in the catalog (e.g., 'IT-DPC-SRI/v0.1.0/italian-radar-dpc-sri').",
    )
    parser.add_argument(
        "--data-stage",
        type=str,
        default="source_data",
        help="The root catalog stage to access (currently expects 'source_data').",
    )
    parser.add_argument(
        "--duration",
        type=str,
        required=True,
        help="ISO 8601 duration for the temporal slice (e.g., 'P1D' for 1 day, 'PT12H' for 12 hours).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Optional ISO 8601 start time. If omitted, defaults to the first timestamp in the dataset.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./data",
        help="Local directory to mirror the downloaded dataset into (default: './data').",
    )

    args = parser.parse_args()

    logger.info("Opening mlcast_datasets root catalog...")
    cat = mlcast_datasets.open_catalog()

    if args.data_stage != "source_data":
        logger.error(f"Currently, only the 'source_data' stage is supported. Got: '{args.data_stage}'")
        sys.exit(1)

    # NOTE: Eventually, we expect the mlcast-datasets catalog to change to include
    # the `data_stage` (e.g., 'source_data', 'processed_data') at the top of the tree.
    # For now, the catalog is flat, so we validate `--data-stage` but do not use it
    # for catalog traversal.
    node = cat

    # Traverse the nested item hierarchy
    # Intake supports dot-notation natively in __getitem__, but we also support slashes
    parts = args.item.replace("/", ".").split(".")
    for part in parts:
        if part not in node:
            logger.error(f"Item part '{part}' not found in current catalog node. Available items: {list(node)}")
            sys.exit(1)
        node = node[part]

    try:
        local_path = get_local_mirrored_path(node, args.base_dir)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if local_path.exists():
        logger.warning(f"Local path already exists: {local_path}. Data may be overwritten or appended.")

    logger.info(f"Loading remote dataset metadata from {args.item}...")
    try:
        ds = node.to_dask()
    except Exception as e:
        logger.error(f"Failed to load dataset via Intake: {e}")
        sys.exit(1)

    if "time" not in ds.coords:
        logger.error("The dataset does not contain a 'time' coordinate for slicing.")
        sys.exit(1)

    # Determine slicing bounds
    if args.start_time is not None:
        start = pd.to_datetime(args.start_time)
    else:
        start = pd.to_datetime(ds.time.min().values)
        logger.info(f"No --start-time provided. Defaulting to dataset start: {start}")

    try:
        delta = pd.Timedelta(args.duration)
    except ValueError as e:
        logger.error(f"Failed to parse duration '{args.duration}': {e}")
        sys.exit(1)

    end = start + delta
    logger.info(f"Slicing dataset from {start} to {end}...")

    ds_subset = ds.sel(time=slice(start, end))

    num_steps = len(ds_subset.time)
    if num_steps == 0:
        logger.error("The requested temporal slice contains zero timesteps. Adjust your start-time or duration.")
        sys.exit(1)

    logger.info(f"Subset contains {num_steps} timesteps. Preparing for download...")

    # Clear encoding to prevent Zarr v3 chunking/codec bugs when writing locally
    for var in ds_subset.variables:
        ds_subset[var].encoding.clear()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading and writing to {local_path} (Zarr format 2)...")
    try:
        from dask.diagnostics import ProgressBar

        with ProgressBar():
            ds_subset.to_zarr(local_path, zarr_format=2, mode="w")
        logger.success(f"Successfully mirrored dataset subset to {local_path}")
    except Exception as e:
        logger.error(f"Failed to write dataset to local Zarr store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

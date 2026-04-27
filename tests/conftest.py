from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture(scope="session")
def fp_test_dataset() -> Path:
    """Download and cache the Italian DPC dataset, returning the local zarr path.

    Returns the path rather than an open dataset so that each test (and each
    dataloader worker) can open the store independently, avoiding shared-state
    issues across processes.
    """
    cache_path = Path(".pytest_cache/italian_dataset_v0.1.0_100t.zarr")

    if not cache_path.exists():
        url = "s3://mlcast-source-datasets/IT-DPC-SRI/v0.1.0/italian-radar-dpc-sri.zarr/"
        storage_options = {
            "anon": True,
            "client_kwargs": {
                "endpoint_url": "https://object-store.os-api.cci2.ecmwf.int",
                "verify": False,
            },
            "config_kwargs": {"signature_version": "s3v4"},
        }

        print(f"\nDownloading dataset to local cache at {cache_path}...")
        ds = xr.open_zarr(url, storage_options=storage_options)

        # Restrict to the first 100 timesteps to ensure random sampling stays within a cached bound
        ds = ds.isel(time=slice(0, 100))

        # Clear encoding to avoid Zarr 3 codec conflicts when writing
        for v in ds.variables:
            ds[v].encoding.clear()

        ds.to_zarr(cache_path, zarr_format=2)
        print("Download and cache complete.")

    return cache_path

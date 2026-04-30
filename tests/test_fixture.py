from pathlib import Path

import xarray as xr


def test_italian_dataset(fp_test_dataset: Path) -> None:
    ds = xr.open_zarr(fp_test_dataset)
    print("Dataset dimensions:", ds.dims)
    print("Variables:", list(ds.data_vars))

    # Actually trigger the read of the first chunk to ensure cache gets filled
    print(ds.isel(time=0).compute())

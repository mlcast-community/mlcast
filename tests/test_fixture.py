def test_italian_dataset(italian_dataset):
    print("Dataset dimensions:", italian_dataset.dims)
    print("Variables:", list(italian_dataset.data_vars))

    # Actually trigger the read of the first chunk to ensure cache gets filled
    print(italian_dataset.isel(time=0).compute())

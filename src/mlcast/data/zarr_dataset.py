"""Contains the Torch dataset for MLCast."""

from pathlib import Path
from random import randint

import numpy as np
import zarr
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    """Torch dataset for MLCast.

    Args:
    ----
    dataset_dir: Path to the dataset.
    dataset_size: Size of the dataset to load.
    input_size: Size of the input to load.
    output_size: Size of the output to load.
    crop_size: Size of the crop to load.

    """

    def __init__(
        self,
        dataset_dir: Path | None = None,
        variable_name: str | None = None,
        input_size: int = 10,
        output_size: int = 5,
        crop_size: int = 256,
    ) -> None:
        """Initialize the Dataset."""
        super().__init__()
        self.dataset_dir: Path | None = dataset_dir

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.seq_len: int = self.input_size + self.output_size

        self.crop_size: int = crop_size
        self.variable_name: str = variable_name

        self.data = zarr.open(self.dataset_dir)

        if self.dataset_dir is None:
            msg = "dataset_dir must be specified"
            raise ValueError(msg)
        if self.variable_name is None:
            msg = "variable_name must be specified"
            raise ValueError(msg)
        self.img_size = self.data[self.variable_name].shape[1:]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.data["time"].shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the item at the given index."""
        sequence = self.data[self.variable_name][index : index + self.seq_len]
        input_seq = sequence[: self.input_size]
        target_seq = sequence[self.input_size :]

        if self.crop_size is not None:
            slice_x = random_slice(self.crop_size, self.img_size[0])
            slice_y = random_slice(self.crop_size, self.img_size[1])
        else:
            slice_x = slice(None, None)
            slice_y = slice(None, None)
        input_seq = input_seq[:, slice_x, slice_y]
        target_seq = target_seq[:, slice_x, slice_y]
        input_seq = np.expand_dims(input_seq, axis=1)
        target_seq = np.expand_dims(target_seq, axis=1)
        return input_seq, target_seq


def random_slice(
    crop: int,
    img_size: int,
    *,
    centered: bool = False,
) -> slice:
    """Randomly crop into an image."""
    if crop > img_size:
        msg = f"""Crop size can't be bigger than the original image.
        ${crop} > ${img_size}"""
        raise ValueError(msg)
    start = img_size / 2 - crop / 2 if centered else randint(0, img_size - crop)

    return slice(start, start + crop)

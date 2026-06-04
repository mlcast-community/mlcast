"""Reconstruction datasets built from sequence datasets.

The reconstruction task reuses normalized source-data sequences and exposes all
overlapping temporal windows of length ``input_steps`` as individual training
samples.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    """Wrap a sequence dataset for stage-1 reconstruction training.

    Parameters
    ----------
    base_sequence_dataset : Dataset
        Dataset whose samples are normalized sequence tensors with shape
        ``(sequence_steps, channels, height, width)``.
    input_steps : int
        Temporal window length to expose for each reconstruction sample.

    Notes
    -----
    Each base sequence contributes all overlapping windows of length
    ``input_steps``. The reconstruction training module is responsible for
    reusing the returned tensor as both the model input and the reconstruction
    target.
    """

    def __init__(self, base_sequence_dataset: Dataset, input_steps: int) -> None:
        if input_steps < 1:
            raise ValueError(f"input_steps ({input_steps}) must be at least 1.")

        self.base_sequence_dataset = base_sequence_dataset
        self.input_steps = input_steps

        sequence_steps = getattr(base_sequence_dataset, "sequence_steps", None)
        if sequence_steps is None:
            raise AttributeError("base_sequence_dataset must expose a 'sequence_steps' attribute.")
        if input_steps > sequence_steps:
            raise ValueError(
                "ReconstructionDataset requires input_steps to be less than or equal to sequence_steps; "
                f"got input_steps={input_steps}, sequence_steps={sequence_steps}."
            )

        self.sequence_steps = sequence_steps
        self.windows_per_sequence = self.sequence_steps - self.input_steps + 1

    def __len__(self) -> int:
        """Return the number of available reconstruction windows."""
        return len(self.base_sequence_dataset) * self.windows_per_sequence

    def __getitem__(self, idx: int) -> Float[Tensor, "input_steps channels height width"]:
        """Return one overlapping reconstruction window.

        Parameters
        ----------
        idx : int
            Flat reconstruction-sample index.

        Returns
        -------
        Float[Tensor, "input_steps channels height width"]
            Window extracted from the wrapped sequence sample.
        """
        sequence_idx = idx // self.windows_per_sequence
        window_start = idx % self.windows_per_sequence
        sequence = self.base_sequence_dataset[sequence_idx]
        window = sequence[window_start : window_start + self.input_steps]
        return torch.nan_to_num(window, nan=-1.0).to(dtype=torch.float32)

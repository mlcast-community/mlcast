"""Simple training script for MLCast."""

from pathlib import Path

import fire
from pytorch_lightning import Trainer
from torch import nn

from mlcast.data.zarr_datamodule import ZarrDataModule
from mlcast.models.base import NowcastingLightningModule
from mlcast.modules import ConvGRU


def main(
    dataset_dir: str,
    variable_name: str,
) -> None:
    """Sample training script for MLCast."""
    model = NowcastingLightningModule(
        net=ConvGRU(),
        loss=nn.MSELoss(),
    )
    trainer = Trainer()
    radklim = ZarrDataModule(
        dataset_dir=Path(
            dataset_dir,
        ),
        variable_name=variable_name,
    )
    trainer.fit(model, datamodule=radklim)


if __name__ == "__main__":
    fire.Fire(main)

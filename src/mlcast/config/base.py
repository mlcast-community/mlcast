"""Base Fiddle experiment definition for radar nowcasting.

This module defines the ``Experiment`` dataclass used across all experiment
configurations.
"""

from dataclasses import dataclass

import pytorch_lightning as pl


@dataclass
class Experiment:
    """Container for Lightning module, data module, and trainer."""

    pl_module: pl.LightningModule
    data: pl.LightningDataModule
    trainer: pl.Trainer

    def run(self) -> None:
        """Train and evaluate the configured model."""
        self.trainer.fit(self.pl_module, datamodule=self.data)
        self.trainer.test(self.pl_module, datamodule=self.data)

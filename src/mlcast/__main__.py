"""Entry point for ``python -m mlcast``.

Uses Fiddle's absl_flags integration to allow overriding any config
parameter from the command line.

Usage examples::

    # Train with default config (must override data paths):
    python -m mlcast \\
        --config config:convgru_experiment \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv \\
        --config set:data.variable_name=RR

    # Override training parameters:
    python -m mlcast \\
        --config config:convgru_experiment \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv \\
        --config set:data.batch_size=32 \\
        --config set:data.num_workers=16 \\
        --config set:pl_module.num_blocks=4 \\
        --config set:pl_module.ensemble_size=4 \\
        --config set:trainer.max_epochs=50
"""

import sys

from absl import app
from fiddle import absl_flags

from . import configs  # noqa: F401 — module must be importable for absl_flags

_CONFIG = absl_flags.DEFINE_fiddle_config(
    "config",
    default_module=sys.modules[f"{__package__}.configs"],
    help_string="Experiment configuration. Use --config config:convgru_experiment to load defaults.",
)


def main(argv: list[str]) -> None:
    """Run training with the Fiddle config from command-line flags."""
    del argv
    from .configs import train_from_config

    train_from_config(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)

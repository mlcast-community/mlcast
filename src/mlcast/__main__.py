"""Entry point for ``python -m mlcast <command>``.

Uses Fiddle's absl_flags integration to allow overriding any config
parameter from the command line.

Usage examples::

    # Train with default config:
    python -m mlcast train \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv

    # Override training parameters:
    python -m mlcast train \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv \\
        --config set:data.batch_size=32 \\
        --config set:pl_module.num_blocks=4 \\
        --config set:trainer.max_epochs=50
"""

import argparse
import sys

from absl import app
from fiddle import absl_flags

from . import configs  # noqa: F401 — module must be importable for absl_flags

_CONFIG = absl_flags.DEFINE_fiddle_config(
    "config",
    default_module=sys.modules[f"{__package__}.configs"],
    help_string="Experiment configuration. Default is training_experiment.",
)


def train_main(argv: list[str]) -> None:
    """Build and run the training experiment from the Fiddle configuration."""
    from .configs import train_from_config

    if _CONFIG.value is None:
        print("Error: --config flag is required.", file=sys.stderr)
        sys.exit(1)
    train_from_config(_CONFIG.value)


def cli() -> None:
    """Console script entry point for ``mlcast`` command."""
    parser = argparse.ArgumentParser(prog="mlcast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train a model.")

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        has_base_config = False
        for i, arg in enumerate(remaining):
            if arg.startswith("--config=config:"):
                has_base_config = True
                break
            if arg == "--config" and i + 1 < len(remaining) and remaining[i + 1].startswith("config:"):
                has_base_config = True
                break

        if not has_base_config:
            remaining = ["--config=config:training_experiment"] + remaining

        app.run(train_main, argv=[sys.argv[0]] + remaining)


if __name__ == "__main__":
    cli()

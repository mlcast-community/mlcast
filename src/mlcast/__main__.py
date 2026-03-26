"""Entry point for ``python -m mlcast <command>``.

Uses Fiddle's absl_flags integration to allow overriding any config
parameter from the command line.

Usage examples::

    # Train with default config:
    python -m mlcast train \\
        --config config:convgru_experiment \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv

    # Override training parameters:
    python -m mlcast train \\
        --config config:convgru_experiment \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv \\
        --config set:data.batch_size=32 \\
        --config set:pl_module.num_blocks=4 \\
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

COMMANDS = ["train"]


def main(argv: list[str]) -> None:
    """Dispatch to the requested subcommand."""
    # argv[0] is the program name, argv[1:] are remaining positional args
    remaining = argv[1:]

    if not remaining:
        print(f"Usage: mlcast <command> [flags]\n\nAvailable commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    command = remaining[0]

    if command == "train":
        from .configs import train_from_config

        if _CONFIG.value is None:
            print("Error: --config flag is required. Example: --config config:convgru_experiment")
            sys.exit(1)
        train_from_config(_CONFIG.value)
    else:
        print(f"Unknown command: {command}\nAvailable commands: {', '.join(COMMANDS)}")
        sys.exit(1)


def cli() -> None:
    """Console script entry point for ``mlcast`` command."""
    app.run(main)


if __name__ == "__main__":
    cli()

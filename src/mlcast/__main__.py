"""Entry point for ``python -m mlcast <command>``.

Uses Fiddle's absl_flags integration to allow overriding any config
parameter from the command line.

Usage examples::

    # Train with default config:
    python -m mlcast train \\
        --config set:data.dataset_factory.zarr_path="'/path/to/data.zarr'" \\
        --config set:data.dataset_factory.csv_path="'/path/to/sampled.csv'"

    # Override training parameters:
    python -m mlcast train \\
        --config set:data.dataset_factory.zarr_path="'/path/to/data.zarr'" \\
        --config set:data.dataset_factory.csv_path="'/path/to/sampled.csv'" \\
        --config set:data.batch_size=32 \\
        --config set:pl_module.network.num_blocks=4 \\
        --config set:trainer.max_epochs=50
"""

import argparse
import ast
import sys

import fiddle as fdl
import torch
from absl import app, flags
from fiddle import absl_flags

from . import config  # noqa: F401 — module must be importable for absl_flags
from .config import train_from_config, training_experiment

FLAGS = flags.FLAGS

_config = absl_flags.DEFINE_fiddle_config(
    "config",
    default_module=config,
    help_string="Experiment configuration. Default is training_experiment.",
)


def get_cli_examples(cfg: fdl.Buildable) -> list[tuple[str, str]]:
    """Returns a list of (description, flag_string) tuples for CLI parameter overrides."""
    return [
        (
            f"Override data layer properties (default batch_size: {cfg.data.batch_size})",
            f"--config set:data.batch_size={max(1, cfg.data.batch_size * 2)}",
        ),
        (
            f"Override the path to the Zarr dataset (default: {cfg.data.dataset_factory.zarr_path})",
            "--config set:data.dataset_factory.zarr_path='/new/path/to/radar.zarr'",
        ),
        (
            f"Override trainer properties (default max_epochs: {cfg.trainer.max_epochs})",
            f"--config set:trainer.max_epochs={max(1, cfg.trainer.max_epochs // 2)}",
        ),
        (
            f"Override network architecture properties (default num_blocks: {cfg.pl_module.network.num_blocks})",
            f"--config set:pl_module.network.num_blocks={max(1, cfg.pl_module.network.num_blocks - 1)}",
        ),
        (
            f"Override the optimizer learning rate (default lr: {cfg.pl_module.optimizer.lr})",
            "--config set:pl_module.optimizer.lr=0.1",
        ),
    ]


def get_fiddler_examples() -> list[tuple[str, str]]:
    """Returns a list of (description, flag_string) tuples for Fiddler mutators."""
    return [
        (
            "Switch to the random sampling dataset (instead of the precomputed CSV sampler)",
            "--config fiddler:use_random_sampler",
        ),
        (
            "Change the input variables and automatically adjust the network's input_channels",
            "--config \"fiddler:set_variables(standard_names=['rainfall_rate', 'reflectivity'])\"",
        ),
        (
            "Toggle whether the loss function ignores masked/invalid pixels",
            '--config "fiddler:toggle_masking(enabled=False)"',
        ),
        (
            "Train using an anonymous S3 object store dataset (e.g. the Italian dataset)",
            '--config "fiddler:use_anon_s3_dataset('
            "zarr_path='s3://mlcast-source-datasets/IT-DPC-SRI/v0.1.0/italian-radar-dpc-sri.zarr/', "
            "endpoint_url='https://object-store.os-api.cci2.ecmwf.int')\"",
        ),
    ]


def _build_help_text(cfg: fdl.Buildable) -> str:
    lines = [
        "Train a model using a Fiddle configuration.",
        "",
        "You can override parameters from the command line using the `--config set:path.to.param=value` syntax.",
        "",
        "Examples (based on the default `training_experiment` config):",
    ]
    for desc, cmd in get_cli_examples(cfg):
        lines.append(f"\n  # {desc}:\n  mlcast train {cmd}")

    lines.append(
        "\nYou can also apply semantic mutators (Fiddlers) to safely change multiple synchronized parameters at once:"
    )

    for desc, cmd in get_fiddler_examples():
        lines.append(f"\n  # {desc}:\n  mlcast train {cmd}")

    lines.append(
        "\nSwitching experiments:\n"
        "  # Switch to a completely different base config function (if defined):\n"
        "  mlcast train --config=config:another_experiment_function\n"
    )
    return "\n".join(lines)


def auto_quote_fiddle_strings(remaining: list[str]) -> list[str]:
    """Auto-quotes unquoted string values in Fiddle set: overrides.

    Fiddle uses `ast.literal_eval` to parse CLI override values, requiring string
    values to be explicitly quoted (e.g. `--config set:a.b="'string'"`). This function
    detects values that fail `ast.literal_eval` and wraps them in single quotes
    automatically, so users can pass bare strings without double-quoting.
    """
    remaining = list(remaining)
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        key = None
        val = None
        form = None  # "split" or "inline"

        # Match `--config set:a.b=c`
        if arg == "--config" and i + 1 < len(remaining) and remaining[i + 1].startswith("set:"):
            set_arg = remaining[i + 1]
            parts = set_arg.split("=", 1)
            if len(parts) == 2:
                key, val = parts[0], parts[1]
                form = "split"
            i += 2

        # Match `--config=set:a.b=c`
        elif arg.startswith("--config=set:"):
            parts = arg.split("=", 2)
            if len(parts) == 3:
                key, val = parts[1], parts[2]
                form = "inline"
            i += 1
        else:
            i += 1
            continue

        if val is not None:
            try:
                ast.literal_eval(val)
            except (ValueError, SyntaxError):
                quoted_val = f"'{val}'"
                if form == "split":
                    remaining[i - 1] = f"{key}={quoted_val}"
                else:
                    remaining[i - 1] = f"--config={key}={quoted_val}"

    return remaining


def train_main(argv: list[str]) -> None:
    """Main training entry point for absl.

    This function is called by `absl.app.run` to ensure that Fiddle configuration
    flags are fully parsed and initialized before starting the training process.

    Parameters
    ----------
    argv : list of str
        The list of command-line arguments passed by absl.
    """

    # Catch legacy Fiddle flags and guide the user to the correct syntax.
    legacy_flags_used = []
    for legacy_flag in ["fiddler", "fdl_set", "fdl_config", "fdl_tags_set"]:
        if legacy_flag in FLAGS and FLAGS[legacy_flag].present:
            legacy_flags_used.append(f"--{legacy_flag}")

    if legacy_flags_used:
        print(
            f"Error: You used legacy Fiddle flags: {', '.join(legacy_flags_used)}.\n"
            "Because this project uses the new Fiddle configuration API, these flags "
            "are silently ignored.\n\n"
            "Please use the nested `--config` flag instead. For example:\n"
            "  Instead of: --fdl_set data.batch_size=32\n"
            "  Use:        --config set:data.batch_size=32\n\n"
            "  Instead of: --fiddler use_random_sampler\n"
            "  Use:        --config fiddler:use_random_sampler\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if _config.value is None:
        print("Error: --config flag is required.", file=sys.stderr)
        sys.exit(1)
    torch.set_float32_matmul_precision("high")
    train_from_config(_config.value)


def cli() -> None:
    """Console script entry point for the ``mlcast`` command.

    This parses standard CLI arguments via `argparse`, injects Fiddle default
    overrides if no base configuration is provided, formats the `--help`
    output, and safely passes execution over to `absl.app.run`.
    """

    # Dynamically generate help text showing Fiddle overrides
    try:
        cfg = training_experiment.as_buildable()
        description_text = _build_help_text(cfg)
    except Exception:
        # Fallback if config generation fails during CLI initialization
        description_text = "Train a model. Overrides can be passed via --config set:key=value"

    parser = argparse.ArgumentParser(
        prog="mlcast",
        description="Entry point for mlcast. Uses Fiddle's absl_flags integration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "train",
        help="Train a model. Overrides can be passed via --config set:key=value",
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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

        # Auto-quote any unquoted string values intended for Fiddle
        remaining = auto_quote_fiddle_strings(remaining)

        app.run(train_main, argv=[sys.argv[0]] + remaining)


if __name__ == "__main__":
    cli()

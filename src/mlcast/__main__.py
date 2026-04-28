"""Entry point for ``python -m mlcast <command>``.

Uses Fiddle's absl_flags integration to allow overriding any config
parameter from the command line.

Usage examples::

    # Train with default config and override dataset path:
    python -m mlcast train \\
        --config fiddler:use_random_sampler \\
        --config set:data.dataset_factory.zarr_path="'/path/to/data.zarr'"

    # Train from a previously saved YAML config:
    python -m mlcast train --config /path/to/config.yaml

    # Train from a saved YAML config with additional overrides:
    python -m mlcast train \\
        --config /path/to/config.yaml \\
        --config set:trainer.max_epochs=50

    # Switch to a different base config function entirely:
    python -m mlcast train --config=config:another_experiment_function
"""

import argparse
import ast
import sys

import fiddle as fdl
import torch
from absl import app, flags
from fiddle import absl_flags
from rich import print as rprint
from rich.console import Console
from rich.text import Text

from . import config  # noqa: F401 — module must be importable for absl_flags
from .config import load_yaml_config, train_from_config, training_experiment

FLAGS = flags.FLAGS

_config = absl_flags.DEFINE_fiddle_config(
    "config",
    default_module=config,
    help_string="Experiment configuration. Default is training_experiment.",
)

flags.DEFINE_boolean(
    "print_config_and_exit",
    False,
    "Print the resolved experiment config and exit without training.",
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


def _build_help_text(cfg: fdl.Buildable) -> Text:
    """Build Rich-highlighted help text for the ``train`` subcommand."""
    t = Text()

    t.append("Train a model using a Fiddle configuration.\n\n", style="bold")
    t.append("You can override parameters from the command line using the ")
    t.append("--config set:path.to.param=value", style="bold cyan")
    t.append(" syntax.\n\n")
    t.append("Examples", style="bold yellow")
    t.append(" (based on the default ")
    t.append("training_experiment", style="bold green")
    t.append(" config):\n")

    for desc, cmd in get_cli_examples(cfg):
        t.append(f"\n  # {desc}:\n", style="dim")
        t.append("  mlcast train ", style="bold")
        t.append(cmd, style="cyan")
        t.append("\n")

    t.append("\nFiddlers", style="bold yellow")
    t.append(" — semantic mutators that safely change multiple synchronised parameters at once:\n")

    for desc, cmd in get_fiddler_examples():
        t.append(f"\n  # {desc}:\n", style="dim")
        t.append("  mlcast train ", style="bold")
        t.append(cmd, style="cyan")
        t.append("\n")

    t.append("\nSwitching experiments:\n", style="bold yellow")
    t.append("  # Switch to a completely different base config function (if defined):\n", style="dim")
    t.append("  mlcast train ", style="bold")
    t.append("--config=config:another_experiment_function\n", style="cyan")
    t.append("\n  # Resume from or reproduce a previously saved YAML config:\n", style="dim")
    t.append("  mlcast train ", style="bold")
    t.append("--config /path/to/config.yaml\n", style="cyan")
    t.append("\n  # Load a YAML config and apply additional overrides on top:\n", style="dim")
    t.append("  mlcast train ", style="bold")
    t.append("--config /path/to/config.yaml --config set:trainer.max_epochs=50\n", style="cyan")

    t.append("\nInspecting the resolved config:\n", style="bold yellow")
    t.append("  # Print the fully resolved config as YAML without starting training:\n", style="dim")
    t.append("  mlcast train ", style="bold")
    t.append("--config fiddler:use_random_sampler --print_config_and_exit\n", style="cyan")

    return t


class _RichHelpParser(argparse.ArgumentParser):
    """ArgumentParser that renders the description with Rich when ``--help`` is requested."""

    _rich_description: Text | None = None

    def print_help(self, file=None) -> None:  # type: ignore[override]
        console = Console(file=file or sys.stdout)
        # Print usage line first (plain argparse)
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        console.print(formatter.format_help(), end="")
        # Rich description
        if self._rich_description is not None:
            console.print(self._rich_description)
        else:
            console.print(self.description or "")
        # Standard options section
        formatter2 = self._get_formatter()
        for action_group in self._action_groups:
            formatter2.start_section(action_group.title)
            formatter2.add_arguments(action_group._group_actions)
            formatter2.end_section()
        console.print(formatter2.format_help(), end="")


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


def _extract_yaml_config_path(remaining: list[str]) -> tuple[str | None, list[str]]:
    """Scan ``remaining`` for a YAML file path passed via ``--config``.

    Detects both ``--config path.yaml`` and ``--config=path.yaml`` forms,
    removes the matching tokens, and returns the path alongside the cleaned
    flag list.  Returns ``(None, remaining)`` unchanged if no YAML path is
    found.

    Parameters
    ----------
    remaining : list of str
        The unparsed CLI tokens after argparse has consumed the subcommand.

    Returns
    -------
    yaml_path : str or None
        The YAML file path if found, otherwise ``None``.
    remaining : list of str
        The flag list with the YAML ``--config`` tokens removed.
    """
    remaining = list(remaining)
    for i, arg in enumerate(remaining):
        if arg == "--config" and i + 1 < len(remaining):
            val = remaining[i + 1]
            if val.endswith(".yaml") or val.endswith(".yml"):
                return val, remaining[:i] + remaining[i + 2 :]
        elif arg.startswith("--config="):
            val = arg[len("--config=") :]
            if val.endswith(".yaml") or val.endswith(".yml"):
                return val, remaining[:i] + remaining[i + 1 :]
    return None, remaining


def _seed_fiddle_flag_from_yaml(yaml_path: str) -> None:
    """Load a YAML config file and seed the Fiddle flag's internal state.

    This replicates what ``FiddleFlag._parse_config()`` does when it handles a
    ``config:`` directive, allowing subsequent ``set:`` and ``fiddler:`` flags
    in ``remaining`` to be applied on top by Fiddle's own flag machinery
    without any special handling on our part.

    ``FLAGS['config']`` gives us the underlying ``FiddleFlag`` instance
    (``fiddle._src.absl_flags.flags.FiddleFlag``), bypassing the
    ``FlagHolder`` wrapper returned by ``DEFINE_fiddle_config``.  Three private
    attributes are set directly:

    - ``_value``: the base ``fdl.Config`` object
    - ``first_command``: marks that a base config has been provided
    - ``_initial_config_expression``: the path, used in error messages

    Parameters
    ----------
    yaml_path : str
        Path to the YAML config file to load.
    """
    cfg = load_yaml_config(yaml_path)
    FLAGS["config"]._value = cfg
    FLAGS["config"].first_command = "config"
    FLAGS["config"]._initial_config_expression = yaml_path


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

    if FLAGS.print_config_and_exit:
        rprint(_config.value)
        sys.exit(0)

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

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model. Overrides can be passed via --config set:key=value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    # Swap in our Rich-aware parser class and attach the highlighted description
    train_parser.__class__ = _RichHelpParser
    if isinstance(description_text, Text):
        train_parser._rich_description = description_text  # type: ignore[attr-defined]
    train_parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this message and exit."
    )

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        # Case 1: user supplied a YAML file path as the base config
        #   e.g. --config /path/to/config.yaml
        #   Extract it from remaining; any set:/fiddler: flags that follow are
        #   applied on top by Fiddle's flag machinery as normal
        yaml_path, remaining = _extract_yaml_config_path(remaining)

        # Case 2: user supplied an explicit base config function
        #   e.g. --config=config:another_experiment_function
        has_explicit_config = any(
            arg.startswith("--config=config:")
            or (arg == "--config" and i + 1 < len(remaining) and remaining[i + 1].startswith("config:"))
            for i, arg in enumerate(remaining)
        )

        # Case 3: no base config from either source — fall back to training_experiment
        if not has_explicit_config and yaml_path is None:
            remaining = ["--config=config:training_experiment"] + remaining

        remaining = auto_quote_fiddle_strings(remaining)

        if yaml_path is not None:
            _seed_fiddle_flag_from_yaml(yaml_path)

        app.run(train_main, argv=[sys.argv[0]] + remaining)


if __name__ == "__main__":
    cli()

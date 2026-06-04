import shlex
import subprocess
import sys

from mlcast.__main__ import get_cli_examples, get_fiddler_examples, get_included_config_names
from mlcast.config import convgru_training_experiment


def test_cli_examples_parse_correctly() -> None:
    """Verify that every CLI override example given in the help text successfully parses."""
    cfg = convgru_training_experiment.as_buildable()
    examples = get_cli_examples(cfg) + get_fiddler_examples()

    for _desc, cmd in examples:
        args = shlex.split(cmd)

        process_args = [sys.executable, "-m", "mlcast", "train"] + args + ["--print_config_and_exit"]

        result = subprocess.run(process_args, capture_output=True, text=True)

        assert result.returncode == 0, f"Command '{cmd}' failed to parse:\n{result.stderr}\n{result.stdout}"


def test_cli_requires_explicit_config() -> None:
    """Train command should fail fast when no base config is provided."""
    result = subprocess.run(
        [sys.executable, "-m", "mlcast", "train"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "base config is required" in result.stderr


def test_cli_help_lists_included_configs() -> None:
    """Help text should advertise the built-in config entry points."""
    result = subprocess.run(
        [sys.executable, "-m", "mlcast", "train", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for name in get_included_config_names():
        assert name in result.stdout

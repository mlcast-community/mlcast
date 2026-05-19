import shlex
import subprocess
import sys

from mlcast.__main__ import get_cli_examples, get_fiddler_examples
from mlcast.config import training_experiment


def test_cli_examples_parse_correctly():
    """Verify that every CLI override example given in the help text successfully parses."""
    cfg = training_experiment.as_buildable()
    examples = get_cli_examples(cfg) + get_fiddler_examples()

    for _desc, cmd in examples:
        # Strip out the leading `mlcast train ` if we used it, but cmd is just `--config ...`
        args = shlex.split(cmd)

        # We can run the __main__.py module using subprocess to ensure isolated absl flag parsing
        process_args = [sys.executable, "-m", "mlcast", "train"] + args + ["--only_check_args"]

        result = subprocess.run(process_args, capture_output=True, text=True)

        # absl prints "unknown flag: --only_check_args" in some versions, or handles it?
        # Let's check what it does.
        assert result.returncode == 0, f"Command '{cmd}' failed to parse:\n{result.stderr}\n{result.stdout}"

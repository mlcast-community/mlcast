import subprocess
import sys
from pathlib import Path

from fiddle._src.experimental.yaml_serialization import dump_yaml

from mlcast.config import training_experiment
from mlcast.config.fiddlers import use_random_sampler


def test_cli_train_command(fp_test_dataset: Path, tmp_path: Path) -> None:
    """Test that the end-to-end training pipeline works via the CLI.

    This ensures that the dynamically generated configs, semantic mutators
    (like `use_random_sampler`), and overridden standard names all play
    nicely together under the `fast_dev_run` flag.
    """
    cmd = [
        sys.executable,
        "-m",
        "mlcast",
        "train",
        "--config",
        "fiddler:use_random_sampler",
        "--config",
        f"set:data.dataset_factory.zarr_path='{fp_test_dataset.absolute()}'",
        "--config",
        "set:data.dataset_factory.standard_names=['rainfall_flux']",
        "--config",
        "set:data.splits={'time': {'train': 0.4, 'val': 0.3, 'test': 0.3}}",
        "--config",
        "set:trainer.fast_dev_run=True",
        "--config",
        "set:data.batch_size=1",
        "--config",
        "set:data.num_workers=0",
        "--config",
        f"set:trainer.default_root_dir='{tmp_path}'",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stdout and stderr to help debug if the test fails
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0, f"CLI training failed. Stderr: {result.stderr}"


def test_cli_train_from_yaml_config(fp_test_dataset: Path, tmp_path: Path) -> None:
    """Test that training works when a pre-saved YAML config is passed via --config.

    Builds the base config in-process, applies all fast-dev overrides (including
    the dataset path) before dumping to YAML, so the subprocess call needs no
    additional --config flags. This exercises the pure load-from-YAML path.
    """
    cfg = training_experiment.as_buildable()
    # Switch to random sampler (no CSV required) and use the correct variable name
    use_random_sampler(cfg)
    cfg.data.dataset_factory.standard_names = ["rainfall_flux"]
    cfg.data.dataset_factory.zarr_path = str(fp_test_dataset.absolute())
    cfg.data.splits = {"time": {"train": 0.4, "val": 0.3, "test": 0.3}}
    cfg.trainer.fast_dev_run = True
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.trainer.default_root_dir = str(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(dump_yaml(cfg))

    cmd = [sys.executable, "-m", "mlcast", "train", "--config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0, f"CLI training from YAML config failed.\nStderr: {result.stderr}"

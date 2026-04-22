import subprocess
import sys
from pathlib import Path


def test_cli_train_command(italian_dataset, tmp_path):
    """Test that the end-to-end training pipeline works via the CLI.

    This ensures that the dynamically generated configs, semantic mutators
    (like `use_random_sampler`), and overridden standard names all play
    nicely together under the `fast_dev_run` flag.
    """
    # The `italian_dataset` fixture ensures this cache path exists
    zarr_path = Path(".pytest_cache/italian_dataset_v0.1.0_100t.zarr").absolute()

    cmd = [
        sys.executable,
        "-m",
        "mlcast",
        "train",
        "--config",
        "fiddler:use_random_sampler",
        "--config",
        f"set:data.dataset_factory.zarr_path='{zarr_path}'",
        "--config",
        "set:data.dataset_factory.standard_names=['rainfall_flux']",
        "--config",
        "set:data.train_ratio=0.4",
        "--config",
        "set:data.val_ratio=0.3",  # Test ratio becomes 0.3 (30 steps > 18 required)
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

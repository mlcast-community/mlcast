"""Simple training script for MLCast using Fiddle configuration.

This script shows the programmatic API for configuring and running
experiments. For CLI usage with arbitrary overrides, use::

    python -m mlcast train \\
        --config config:convgru_experiment \\
        --config set:data.zarr_path=/path/to/data.zarr \\
        --config set:data.csv_path=/path/to/sampled.csv \\
        --config set:data.batch_size=32 \\
        --config set:trainer.max_epochs=50
"""

import fiddle as fdl

from mlcast.configs import convgru_experiment


def main():
    # Get the config graph — all parameters are overridable via dot-access
    cfg = convgru_experiment.as_buildable(
        zarr_path="/path/to/data.zarr",
        csv_path="/path/to/sampled.csv",
        variable_name="RR",
    )

    # Override any nested parameter before building
    cfg.data.batch_size = 32
    cfg.data.num_workers = 16
    cfg.pl_module.num_blocks = 4
    cfg.pl_module.ensemble_size = 4
    cfg.trainer.max_epochs = 50

    # Build all objects and run training + testing
    experiment = fdl.build(cfg)
    experiment.run()


if __name__ == "__main__":
    main()

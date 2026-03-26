"""Simple training script for MLCast using Fiddle configuration.

Example usage::

    python simple_train.py --zarr-path /path/to/data.zarr \
                           --csv-path /path/to/sampled.csv \
                           --variable-name RR

For advanced customization, modify the config before building::

    cfg = convgru_experiment.as_buildable(zarr_path=..., csv_path=...)
    cfg.pl_module.num_blocks = 4
    cfg.data.batch_size = 32
    cfg.trainer.max_epochs = 50
    fdl.build(cfg).run()
"""

import argparse

import fiddle as fdl

from mlcast.configs import convgru_experiment, train_from_config


def main():
    parser = argparse.ArgumentParser(description="MLCast simple training")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to Zarr dataset")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to sampled datacubes CSV")
    parser.add_argument("--variable-name", type=str, default="RR", help="Variable name in Zarr store")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    args = parser.parse_args()

    cfg = convgru_experiment.as_buildable(
        zarr_path=args.zarr_path,
        csv_path=args.csv_path,
        variable_name=args.variable_name,
    )

    # Apply CLI overrides
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.trainer.max_epochs = args.max_epochs

    train_from_config(cfg)


if __name__ == "__main__":
    main()

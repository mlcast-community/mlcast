"""Entry point for ``python -m mlcast``."""

import argparse

from .configs import convgru_experiment, train_from_config


def main() -> None:
    """Run a training experiment.

    Usage examples::

        # Run with defaults (override zarr_path and csv_path programmatically)
        python -m mlcast

        # Modify config programmatically in a script:
        #   cfg = convgru_experiment.as_buildable(zarr_path=..., csv_path=...)
        #   cfg.pl_module.num_blocks = 4
        #   cfg.data.batch_size = 32
        #   fdl.build(cfg).run()
    """
    parser = argparse.ArgumentParser(description="MLCast training")
    parser.add_argument("--zarr-path", type=str, default="./data/radar.zarr", help="Path to Zarr dataset")
    parser.add_argument("--csv-path", type=str, default="./data/sampled_datacubes.csv", help="Path to sampled CSV")
    parser.add_argument("--variable-name", type=str, default="RR", help="Variable name in Zarr store")
    args = parser.parse_args()

    cfg = convgru_experiment.as_buildable(
        zarr_path=args.zarr_path,
        csv_path=args.csv_path,
        variable_name=args.variable_name,
    )
    train_from_config(cfg)


if __name__ == "__main__":
    main()

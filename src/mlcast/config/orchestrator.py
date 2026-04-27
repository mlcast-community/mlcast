"""High-level configuration orchestrators."""

import re
from pathlib import Path

import fiddle as fdl
import yaml
from fiddle import printing as fdp
from fiddle.experimental.yaml_serialization import dump_yaml

from .consistency_checks import validate_config


# Register a YAML representer for Python types (classes) so they can be serialized
def _type_representer(dumper, data):
    return dumper.represent_scalar("!type", f"{data.__module__}.{data.__qualname__}")


yaml.SafeDumper.add_representer(type, _type_representer)


def train_from_config(cfg: fdl.Config) -> None:
    """Build and run an experiment from a Fiddle config.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration as returned by `training_experiment`.
    """
    validate_config(cfg)

    experiment = fdl.build(cfg)

    # Save config YAML to the trainer's log directory
    if experiment.trainer.log_dir is not None:
        log_dir = Path(experiment.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        config_path = log_dir / "config.yaml"
        config_path.write_text(dump_yaml(cfg))
        print(f"Config saved to {config_path}")
    else:
        print("Config not saved because trainer.log_dir is None (likely fast_dev_run).")

    # Log flattened configuration as hyperparameters if a logger is configured
    if experiment.trainer.logger is not None:
        flat_cfg = fdp.as_dict_flattened(cfg)
        # fdp.as_dict_flattened uses bracket notation for list indices
        # (e.g. "trainer.callbacks[0].monitor"), which MLflow rejects — its
        # parameter key validator forbids "[" and "]". Replace "[n]" with ".n"
        # to produce valid, human-readable keys without losing any information.
        loggable_cfg = {
            re.sub(r"\[(\d+)\]", r".\1", k): v if isinstance(v, int | float | str | bool) else str(v)
            for k, v in flat_cfg.items()
        }
        experiment.trainer.logger.log_hyperparams(loggable_cfg)
        print("Logged flattened Fiddle configuration to trainer.logger")

    experiment.run()

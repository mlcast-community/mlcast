"""Orchestration layer between a built Fiddle config and the Lightning trainer.

This module is the bridge between configuration and execution.  Its two public
responsibilities are:

1. **Validation and building** — ``train_from_config`` calls
   ``validate_config`` to enforce cross-parameter contracts, then calls
   ``fdl.build`` to instantiate all objects, and finally delegates to
   ``experiment.run()``.

2. **Config persistence** — before training starts,
   ``_log_experiment_config_yaml_file`` serialises the Fiddle config to YAML
   and routes it to the right destination based on the active logger:
   - ``TensorBoardLogger``: written alongside the TB event files.
   - ``MLFlowLogger``: uploaded as an MLflow artifact (temp file, then deleted).
   - ``WandbLogger``: uploaded as a W&B artifact; ``artifact.wait()`` blocks
     until the async upload completes.
   - Fallback: written to ``trainer.default_root_dir``.
"""

import re
import tempfile
from pathlib import Path

import fiddle as fdl
import pytorch_lightning as pl
import yaml
from fiddle import printing as fdp
from fiddle.experimental.yaml_serialization import dump_yaml
from pytorch_lightning.loggers import TensorBoardLogger

try:
    from pytorch_lightning.loggers import MLFlowLogger
except ImportError:  # mlflow is an optional dependency
    MLFlowLogger = None  # type: ignore[assignment,misc]

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # wandb is an optional dependency
    wandb = None  # type: ignore[assignment]
    WandbLogger = None  # type: ignore[assignment,misc]

from .consistency_checks import validate_config


# Register a YAML representer for Python types (classes) so they can be serialized
def _type_representer(dumper, data):
    return dumper.represent_scalar("!type", f"{data.__module__}.{data.__qualname__}")


yaml.SafeDumper.add_representer(type, _type_representer)


def _log_experiment_config_yaml_file(cfg: fdl.Config, trainer: pl.Trainer) -> None:
    """Save or upload the Fiddle config YAML, dispatching on the logger type.

    For known remote loggers (MLflow, W&B) the YAML is uploaded as an artifact
    so it is permanently associated with the run on the tracking server — no
    local copy is written.  For TensorBoardLogger it is written alongside the
    TB event files.  For any other logger (or no logger) it falls back to
    ``trainer.default_root_dir``.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to serialise.
    trainer : pl.Trainer
        The built Lightning trainer whose logger determines the destination.

    Raises
    ------
    TypeError
        If a live Python object is present in the config graph instead of a
        ``fdl.Config`` or ``fdl.Partial`` — the offending object is identified
        in the error message.
    """
    try:
        config_yaml = dump_yaml(cfg)
    except yaml.representer.RepresenterError as e:
        raise TypeError(
            "Failed to serialise the Fiddle config to YAML. This usually means a "
            "live Python object was placed directly in the config graph instead of "
            "being wrapped in fdl.Config(...) or fdl.Partial(...). "
            f"Offending object: {e.args[1]!r}"
        ) from e
    logger = trainer.logger

    if isinstance(logger, TensorBoardLogger):
        # TensorBoard sets a versioned local log_dir — save alongside TB logs
        path = Path(logger.log_dir) / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config_yaml)
        print(f"Config saved to {path}")

    elif MLFlowLogger is not None and isinstance(logger, MLFlowLogger):
        # log_artifact requires a real filesystem path, so use a temp file
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            tmp = Path(f.name)
        logger.experiment.log_artifact(logger.run_id, str(tmp), artifact_path="config")
        tmp.unlink()
        print("Config uploaded to MLflow as artifact under 'config/config.yaml'")

    elif WandbLogger is not None and isinstance(logger, WandbLogger):
        artifact = wandb.Artifact(name="config", type="config")
        with artifact.new_file("config.yaml") as f:
            f.write(config_yaml)
        logger.experiment.log_artifact(artifact)
        # wait() blocks until the artifact upload completes; without this the
        # upload is async and may not finish if training crashes shortly after
        artifact.wait()
        print("Config uploaded to W&B as artifact 'config'")

    else:
        # No logger or unrecognised logger type — save locally under default_root_dir
        path = Path(trainer.default_root_dir) / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config_yaml)
        print(f"Config saved locally to {path} (no recognised remote logger)")


def train_from_config(cfg: fdl.Config) -> None:
    """Build and run an experiment from a Fiddle config.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration as returned by `training_experiment`.
    """
    validate_config(cfg)

    experiment = fdl.build(cfg)

    _log_experiment_config_yaml_file(cfg, experiment.trainer)

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

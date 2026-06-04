"""Fiddler mutators for high-level semantic configuration changes.

Fiddlers are functions that accept a ``fdl.Config`` and mutate them in place.
They are the right tool when a change spans multiple config parameters that
must stay in sync — for example, switching the dataset class while preserving
its existing parameters, or enabling masking consistently across both the data
pipeline and the loss function.

Use fiddlers from the CLI via ``--config fiddler:<name>`` or
``--config "fiddler:<name>(arg=value)"``, or call them directly on a buildable
config in Python before passing it to ``fdl.build()``.
"""

import functools
import inspect
import os
from collections.abc import Callable

import fiddle as fdl
import torch.nn as nn
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger

from mlcast.config.base import Experiment

from ..callbacks import LogSystemInfoCallback
from ..data.sequence import SourceDataRandomSequenceDataset


def _iter_experiment_configs(cfg: fdl.Buildable):
    """Yield all ``fdl.Config`` sub-nodes whose callable is ``Experiment``, depth-first.

    Parameters
    ----------
    cfg : fdl.Buildable
        Root of the Fiddle configuration tree to traverse.

    Yields
    ------
    fdl.Config
        Each sub-config whose ``fdl.get_callable`` is the ``Experiment``
        dataclass.
    """
    if not isinstance(cfg, fdl.Buildable):
        return
    try:
        if fdl.get_callable(cfg) is Experiment:
            yield cfg
    except (TypeError, AttributeError):
        pass
    try:
        for child in fdl.ordered_arguments(cfg).values():
            yield from _iter_experiment_configs(child)
    except (TypeError, AttributeError):
        pass


def _find_nn_modules_with_input_channels(cfg: fdl.Buildable):
    """Yield all ``fdl.Config`` nodes for ``nn.Module`` subclasses that accept ``input_channels``.

    Parameters
    ----------
    cfg : fdl.Buildable
        Root of the Fiddle configuration tree to traverse (typically
        ``cfg.pl_module``).

    Yields
    ------
    fdl.Config
        Each sub-config whose callable is an ``nn.Module`` subclass with
        ``input_channels`` in its ``__init__`` signature.
    """
    if not isinstance(cfg, fdl.Config):
        return
    try:
        cls = fdl.get_callable(cfg)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            if "input_channels" in inspect.signature(cls.__init__).parameters:
                yield cfg
    except (TypeError, AttributeError):
        pass
    try:
        for child in fdl.ordered_arguments(cfg).values():
            yield from _find_nn_modules_with_input_channels(child)
    except (TypeError, AttributeError):
        pass


def applies_to_experiments(fiddler: Callable) -> Callable:
    """Decorate a fiddler so it applies to every ``Experiment`` sub-config in the tree.

    This makes fiddlers work with both flat ``Experiment`` configs (returned by
    ``convgru_training_experiment``) and nested containers like
    ``LatentDiffusionTrainingExperiment`` that contain multiple ``Experiment`` instances.

    Parameters
    ----------
    fiddler : Callable
        Fiddler function whose first argument is a ``fdl.Config``.

    Returns
    -------
    Callable
        Wrapped fiddler that traverses the config tree for ``Experiment``
        sub-configs and applies the original fiddler to each one.
    """

    @functools.wraps(fiddler)
    def wrapper(cfg: fdl.Buildable, *args: object, **kwargs: object) -> None:
        experiments = list(_iter_experiment_configs(cfg))
        if experiments:
            for exp_cfg in experiments:
                fiddler(exp_cfg, *args, **kwargs)
        else:
            fiddler(cfg, *args, **kwargs)

    return wrapper


@applies_to_experiments
def set_variables(cfg: fdl.Config, standard_names: list[str]) -> None:
    """Fiddler to synchronize dataset variables with the network's input channels.

    Sets ``sequence_dataset_factory.standard_names`` on the data config and
    walks ``cfg.pl_module`` to find any ``nn.Module`` with an ``input_channels``
    ``__init__`` parameter (e.g. ``ConvGruModel``, ``Encoder``), keeping it in
    sync with the number of loaded variables.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    standard_names : list of str
        The new list of standard names to load.
    """
    cfg.data.sequence_dataset_factory.standard_names = standard_names
    found = False
    for module_cfg in _find_nn_modules_with_input_channels(cfg.pl_module):
        module_cfg.input_channels = len(standard_names)
        found = True
    if not found:
        logger.warning(
            "set_variables: no nn.Module under pl_module has an 'input_channels' parameter; channel count not updated."
        )


@applies_to_experiments
def toggle_masking(cfg: fdl.Config, enabled: bool) -> None:
    """Fiddler to synchronize forecasting-mask yielding with masked loss computation.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    enabled : bool
        Whether to enable masking or not.
    """
    cfg.data.return_mask = enabled
    cfg.pl_module.masked_loss = enabled


@applies_to_experiments
def use_random_sampler(cfg: fdl.Config) -> None:
    """Fiddler to switch the sequence dataset factory to use the random sampler.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    # Keep the existing parameters but change the underlying class
    cfg.data.sequence_dataset_factory = fdl.Partial(
        SourceDataRandomSequenceDataset,
        zarr_path=cfg.data.sequence_dataset_factory.zarr_path,
        standard_names=cfg.data.sequence_dataset_factory.standard_names,
        sequence_steps=cfg.data.sequence_dataset_factory.sequence_steps,
        storage_options=getattr(cfg.data.sequence_dataset_factory, "storage_options", None),
    )


@applies_to_experiments
def use_ratio_splits(cfg: fdl.Config, train: float, val: float) -> None:
    """Fiddler to set fraction-based train/val/test splits on the data module."""
    cfg.data.splits = {"time": {"train": train, "val": val, "test": 1.0 - train - val}}


@applies_to_experiments
def use_anon_s3_dataset(cfg: fdl.Buildable, zarr_path: str, endpoint_url: str) -> None:
    """Configure the dataset factory to read anonymously from an S3 object store.

    Parameters
    ----------
    cfg : fdl.Buildable
        The Fiddle configuration to mutate.
    zarr_path : str
        The S3 URI path to the Zarr dataset (e.g., s3://bucket/path.zarr).
    endpoint_url : str
        The endpoint URL for the S3 object store.
    """
    cfg.data.sequence_dataset_factory.zarr_path = zarr_path
    cfg.data.sequence_dataset_factory.storage_options = {
        "anon": True,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "verify": False,
        },
        "config_kwargs": {"signature_version": "s3v4"},
    }


@applies_to_experiments
def use_mlflow_logger(cfg: fdl.Config) -> None:
    """Fiddler to switch the trainer logger to MLflow.

    Replaces the default TensorBoardLogger with an MLFlowLogger, inheriting
    the experiment name from the existing logger config. The tracking URI and
    run name are left unset, deferring to the ``MLFLOW_TRACKING_URI`` and
    ``MLFLOW_RUN_NAME`` environment variables (or MLflow defaults).

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.warning(
            "MLFLOW_TRACKING_URI is not set. MLflow will log to a local './mlruns' directory. "
            "Set MLFLOW_TRACKING_URI to point to a remote tracking server, "
            "e.g. export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    cfg.trainer.logger = fdl.Config(MLFlowLogger, experiment_name=cfg.trainer.logger.name)
    cfg.trainer.callbacks.append(fdl.Config(LogSystemInfoCallback))

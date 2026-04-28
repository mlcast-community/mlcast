"""Fiddler mutators for high-level semantic configuration changes.

Fiddlers are functions that accept a ``fdl.Config`` and mutate it in place.
They are the right tool when a change spans multiple config parameters that
must stay in sync — for example, switching the dataset class while preserving
its existing parameters, or enabling masking consistently across both the data
pipeline and the loss function.

Use fiddlers from the CLI via ``--config fiddler:<name>`` or
``--config "fiddler:<name>(arg=value)"``, or call them directly on a buildable
config in Python before passing it to ``fdl.build()``.
"""

import inspect
import os

import fiddle as fdl
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger

from ..callbacks import LogSystemInfoCallback
from ..data.source_data_datasets import SourceDataRandomSamplingDataset


def set_variables(cfg: fdl.Config, standard_names: list[str]) -> None:
    """Fiddler to synchronize dataset variables with the network's input channels.

    Sets ``dataset_factory.standard_names`` on the data config and, when the
    network config exposes an ``input_channels`` parameter (e.g.
    ``ConvGruModel``), keeps it in sync.  Networks that use a different
    parameter name for the channel count (e.g. ``HalfUNet`` uses
    ``in_channels``) are left unchanged — callers are responsible for keeping
    that parameter consistent when swapping in an external architecture.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    standard_names : list of str
        The new list of standard names to load.
    """
    cfg.data.dataset_factory.standard_names = standard_names
    network_cls = cfg.pl_module.network.__fn_or_cls__
    sig = inspect.signature(network_cls.__init__)
    if "input_channels" in sig.parameters:
        cfg.pl_module.network.input_channels = len(standard_names)
    else:
        logger.warning(
            "set_variables: network {} has no 'input_channels' parameter; "
            "channel count not updated. Set it manually on the network config.",
            network_cls.__name__,
        )


def toggle_masking(cfg: fdl.Config, enabled: bool) -> None:
    """Fiddler to synchronize dataset mask yielding with masked loss computation.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    enabled : bool
        Whether to enable masking or not.
    """
    cfg.data.dataset_factory.return_mask = enabled
    cfg.pl_module.masked_loss = enabled


def use_random_sampler(cfg: fdl.Config) -> None:
    """Fiddler to switch the dataset factory to use the random sampler.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    """
    # Keep the existing parameters but change the underlying class
    cfg.data.dataset_factory = fdl.Partial(
        SourceDataRandomSamplingDataset,
        zarr_path=cfg.data.dataset_factory.zarr_path,
        standard_names=cfg.data.dataset_factory.standard_names,
        steps=cfg.data.dataset_factory.steps,
        return_mask=cfg.data.dataset_factory.return_mask,
        storage_options=getattr(cfg.data.dataset_factory, "storage_options", None),
    )


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
    cfg.data.dataset_factory.zarr_path = zarr_path
    cfg.data.dataset_factory.storage_options = {
        "anon": True,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "verify": False,
        },
        "config_kwargs": {"signature_version": "s3v4"},
    }


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

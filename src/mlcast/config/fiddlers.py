"""Fiddler mutators for high-level semantic configuration changes."""

import fiddle as fdl

from ..data.source_datasets import SourceDataRandomSamplingDataset


def set_variables(cfg: fdl.Config, standard_names: list[str]) -> None:
    """Fiddler to synchronize dataset variables with the network's input channels.

    Parameters
    ----------
    cfg : fdl.Config
        The Fiddle configuration to mutate.
    standard_names : list of str
        The new list of standard names to load.
    """
    cfg.data.dataset_factory.standard_names = standard_names
    cfg.pl_module.network.input_channels = len(standard_names)


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

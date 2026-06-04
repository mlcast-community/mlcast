"""Cross-parameter validation constraints for Fiddle configurations.

Consistency checks are predicate functions that raise ``ValueError`` when
two or more config parameters that must stay in sync have drifted apart.
Unlike fiddlers (which *mutate* a config to enforce a policy), consistency
checks are *read-only* — they inspect the config and signal problems early,
before ``fdl.build()`` is called and before any heavyweight objects are
instantiated.

Call ``validate_config(cfg)`` explicitly after all fiddlers have been
applied and before handing the config off to the training orchestrator.
"""

import fiddle as fdl
from loguru import logger


def _validate_forecasting_experiment_cfg(cfg: fdl.Config) -> None:
    """Validate a single-stage forecasting experiment configuration.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration for a single forecasting experiment.

    Raises
    ------
    ValueError
        If any configuration contract is violated.
    """
    sequence_dataset_factory = cfg.data.sequence_dataset_factory
    network = cfg.pl_module.network
    pl_module = cfg.pl_module
    data = cfg.data

    # Contract 1: Network input_channels == len(sequence_dataset_factory.standard_names)
    num_vars = len(sequence_dataset_factory.standard_names)
    try:
        net_input_channels = network.input_channels
    except AttributeError:
        logger.warning(
            "Warning: can't ensure network input_channels matches the number of dataset variables, "
            "because network {} doesn't expose 'input_channels'.",
            network.__class__.__name__,
        )
        net_input_channels = None
    if net_input_channels is not None and net_input_channels != num_vars:
        raise ValueError(
            f"Contract 1 violated: network input_channels ({net_input_channels}) "
            f"must equal the number of standard_names ({num_vars})."
        )

    # Contract 2: Sequence dataset width must be divisible by 2 ** network.num_blocks
    try:
        num_blocks = network.num_blocks
    except AttributeError:
        logger.warning(
            "Warning: can't ensure dataset width is compatible with the network downsampling factor, "
            "because network {} doesn't expose 'num_blocks'.",
            network.__class__.__name__,
        )
        num_blocks = None
    if num_blocks is not None:
        width = getattr(sequence_dataset_factory, "width", 256)
        divisor = 2**num_blocks
        if width % divisor != 0:
            raise ValueError(
                f"Contract 2 violated: Dataset width ({width}) must be divisible by "
                f"2 ** network.num_blocks ({divisor})."
            )

    # Contract 3: Ensemble models require CRPS or AFCRPS
    ensemble_size = getattr(network, "ensemble_size", 1)
    if ensemble_size > 1:
        if str(pl_module.loss_class).lower() not in ["crps", "afcrps"]:
            raise ValueError(
                f"Contract 3 violated: Ensemble models (ensemble_size={ensemble_size}) "
                f"require 'crps' or 'afcrps' loss, got '{pl_module.loss_class}'."
            )

    # Contract 4: Forecasting mask return must match model masked_loss
    if bool(data.return_mask) != bool(pl_module.masked_loss):
        raise ValueError(
            f"Contract 4 violated: data.return_mask ({data.return_mask}) "
            f"must match pl_module.masked_loss ({pl_module.masked_loss})."
        )

    # Contract 5: Dataset input_steps must match model input_steps
    try:
        net_input_steps = network.input_steps
    except AttributeError:
        logger.warning(
            "Warning: can't ensure network input_steps matches data.input_steps, "
            "because network {} doesn't expose 'input_steps'.",
            network.__class__.__name__,
        )
        net_input_steps = None
    if net_input_steps is not None and net_input_steps != data.input_steps:
        raise ValueError(
            f"Contract 5 violated: network input_steps ({net_input_steps}) "
            f"must equal data.input_steps ({data.input_steps})."
        )

    # Contract 6: Dataset forecast_steps must match model forecast_steps
    try:
        net_forecast_steps = network.forecast_steps
    except AttributeError:
        logger.warning(
            "Warning: can't ensure network forecast_steps matches data.forecast_steps, "
            "because network {} doesn't expose 'forecast_steps'.",
            network.__class__.__name__,
        )
        net_forecast_steps = None
    if net_forecast_steps is not None and net_forecast_steps != data.forecast_steps:
        raise ValueError(
            f"Contract 6 violated: network forecast_steps ({net_forecast_steps}) "
            f"must equal data.forecast_steps ({data.forecast_steps})."
        )


def _validate_latent_diffusion_experiment_cfg(cfg: fdl.Config) -> None:
    """Validate a two-stage latent diffusion training experiment configuration.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration for a two-stage latent diffusion experiment.

    Raises
    ------
    ValueError
        If any latent-diffusion-specific configuration contract is violated.
    """
    stage1 = cfg.stage1
    stage2 = cfg.stage2

    autoencoder = stage1.pl_module.network
    if autoencoder is not stage2.pl_module.autoencoder:
        raise ValueError(
            "LatentDiffusion contract violated: stage1 and stage2 must share the same autoencoder config object."
        )

    stage1_data = stage1.data
    stage2_data = stage2.data
    if stage1_data.input_steps != stage2_data.input_steps:
        raise ValueError(
            "LatentDiffusion contract violated: stage1 and stage2 must use the same input_steps; "
            f"got {stage1_data.input_steps} and {stage2_data.input_steps}."
        )

    stage2_module = stage2.pl_module
    if stage2_data.forecast_steps != stage2_module.forecast_steps:
        raise ValueError(
            "LatentDiffusion contract violated: stage2 data.forecast_steps must match the latent diffusion "
            f"task module; got {stage2_data.forecast_steps} and {stage2_module.forecast_steps}."
        )

    if len(stage1_data.sequence_dataset_factory.standard_names) != autoencoder.encoder.input_channels:
        raise ValueError(
            "LatentDiffusion contract violated: autoencoder encoder input_channels must match the "
            "number of source variables."
        )


def validate_config(cfg: fdl.Config) -> None:
    """Validate cross-system constraints on a Fiddle configuration before training.

    Parameters
    ----------
    cfg : fdl.Config
        Fiddle configuration.

    Raises
    ------
    ValueError
        If any configuration contract is violated.
    """
    if hasattr(cfg, "stage1") and hasattr(cfg, "stage2"):
        _validate_latent_diffusion_experiment_cfg(cfg)
        return

    _validate_forecasting_experiment_cfg(cfg)

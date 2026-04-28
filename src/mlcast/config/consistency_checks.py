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
    dataset_factory = cfg.data.dataset_factory
    network = cfg.pl_module.network
    pl_module = cfg.pl_module

    # Contract 1: Network input_channels == len(dataset_factory.standard_names)
    # Only enforced when the network config exposes input_channels. Networks
    # using a different parameter name (e.g. HalfUNet uses in_channels) are
    # skipped — the caller is responsible for keeping those in sync.
    num_vars = len(dataset_factory.standard_names)
    try:
        net_input_channels = network.input_channels
    except AttributeError:
        net_input_channels = None
    if net_input_channels is not None and net_input_channels != num_vars:
        raise ValueError(
            f"Contract 1 violated: network input_channels ({net_input_channels}) "
            f"must equal the number of standard_names ({num_vars})."
        )

    # Contract 2: Dataset width must be divisible by 2 ** network.num_blocks
    # Only enforced when the network config exposes num_blocks (e.g. ConvGruModel).
    # External architectures that do not use a power-of-two downsampling factor
    # (e.g. HalfUNet) will not have this parameter and are skipped.
    try:
        num_blocks = network.num_blocks
    except AttributeError:
        num_blocks = None
    if num_blocks is not None:
        width = getattr(dataset_factory, "width", 256)
        divisor = 2**num_blocks
        if width % divisor != 0:
            raise ValueError(
                f"Contract 2 violated: Dataset width ({width}) must be divisible by "
                f"2 ** network.num_blocks ({divisor})."
            )

    # Contract 3: Dataset steps > forecast_steps
    if dataset_factory.steps <= pl_module.forecast_steps:
        raise ValueError(
            f"Contract 3 violated: Dataset steps ({dataset_factory.steps}) "
            f"must be greater than forecast_steps ({pl_module.forecast_steps})."
        )

    # Contract 4: Ensemble models require CRPS or AFCRPS
    if pl_module.ensemble_size > 1:
        if str(pl_module.loss_class).lower() not in ["crps", "afcrps"]:
            raise ValueError(
                f"Contract 4 violated: Ensemble models (ensemble_size={pl_module.ensemble_size}) "
                f"require 'crps' or 'afcrps' loss, got '{pl_module.loss_class}'."
            )

    # Contract 5: Dataset return_mask must match model masked_loss
    if bool(dataset_factory.return_mask) != bool(pl_module.masked_loss):
        raise ValueError(
            f"Contract 5 violated: dataset_factory.return_mask ({dataset_factory.return_mask}) "
            f"must match pl_module.masked_loss ({pl_module.masked_loss})."
        )

import pytest

from mlcast.config import training_experiment, validate_config


def test_contract_1_input_channels():
    """Verify Contract 1: Network input_channels == len(dataset_factory.standard_names)."""
    cfg = training_experiment.as_buildable()
    # Break Contract 1
    cfg.pl_module.network.input_channels = 2
    cfg.data.dataset_factory.standard_names = ["rainfall_rate"]

    with pytest.raises(ValueError, match="Contract 1 violated:"):
        validate_config(cfg)


def test_contract_2_spatial_divisibility():
    """Verify Contract 2: Dataset width must be divisible by 2 \\*\\* network.num_blocks."""
    cfg = training_experiment.as_buildable()
    # Break Contract 2
    cfg.data.dataset_factory.width = 250
    cfg.pl_module.network.num_blocks = 4

    with pytest.raises(ValueError, match="Contract 2 violated:"):
        validate_config(cfg)


def test_contract_3_temporal_sync():
    """Verify Contract 3: Dataset steps > forecast_steps."""
    cfg = training_experiment.as_buildable()
    # Break Contract 3
    cfg.data.dataset_factory.steps = 10
    cfg.pl_module.forecast_steps = 12

    with pytest.raises(ValueError, match="Contract 3 violated:"):
        validate_config(cfg)


def test_contract_4_probabilistic_loss():
    """Verify Contract 4: Ensemble models require CRPS or AFCRPS."""
    cfg = training_experiment.as_buildable()
    # Break Contract 4
    cfg.pl_module.ensemble_size = 5
    cfg.pl_module.loss_class = "mse"

    with pytest.raises(ValueError, match="Contract 4 violated:"):
        validate_config(cfg)


def test_contract_5_masking_sync():
    """Verify Contract 5: Dataset return_mask must match model masked_loss."""
    cfg = training_experiment.as_buildable()
    # Break Contract 5
    cfg.data.dataset_factory.return_mask = True
    cfg.pl_module.masked_loss = False

    with pytest.raises(ValueError, match="Contract 5 violated:"):
        validate_config(cfg)

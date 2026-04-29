import pytest

from mlcast.config import training_experiment, validate_config
from mlcast.data.source_data_datasets import SourceDataPrecomputedSamplingDataset


def test_contract_1_input_channels() -> None:
    """Verify Contract 1: Network input_channels == len(dataset_factory.standard_names)."""
    cfg = training_experiment.as_buildable()
    # Break Contract 1
    cfg.pl_module.network.input_channels = 2
    cfg.data.dataset_factory.standard_names = ["rainfall_rate"]

    with pytest.raises(ValueError, match="Contract 1 violated:"):
        validate_config(cfg)


def test_contract_2_spatial_divisibility() -> None:
    """Verify Contract 2: Dataset width must be divisible by 2 \\*\\* network.num_blocks."""
    cfg = training_experiment.as_buildable()
    # Break Contract 2
    cfg.data.dataset_factory.width = 250
    cfg.pl_module.network.num_blocks = 4

    with pytest.raises(ValueError, match="Contract 2 violated:"):
        validate_config(cfg)


def test_contract_3_probabilistic_loss() -> None:
    """Verify Contract 3: Ensemble models require CRPS or AFCRPS."""
    cfg = training_experiment.as_buildable()
    # Break Contract 3
    cfg.pl_module.ensemble_size = 5
    cfg.pl_module.loss_class = "mse"

    with pytest.raises(ValueError, match="Contract 3 violated:"):
        validate_config(cfg)


def test_contract_4_masking_sync() -> None:
    """Verify Contract 4: Dataset return_mask must match model masked_loss."""
    cfg = training_experiment.as_buildable()
    # Break Contract 4
    cfg.data.dataset_factory.return_mask = True
    cfg.pl_module.masked_loss = False

    with pytest.raises(ValueError, match="Contract 4 violated:"):
        validate_config(cfg)


def test_dataset_forecast_steps_guard() -> None:
    """Verify that dataset raises ValueError when forecast_steps >= steps."""
    with pytest.raises(ValueError, match="forecast_steps"):
        SourceDataPrecomputedSamplingDataset(
            zarr_path="dummy.zarr",
            csv_path="dummy.csv",
            standard_names=["rainfall_rate"],
            steps=5,
            forecast_steps=5,
        )

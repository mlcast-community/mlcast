from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlcast.config import convgru_training_experiment, latent_diffusion_experiment, train_from_config


@patch("mlcast.config.orchestrator.fdl.build")
def test_train_from_config_valid(mock_build: Any, tmp_path: Path) -> None:
    """Verify that a valid configuration passes validation and builds."""
    mock_build.return_value.trainer.log_dir = str(tmp_path)
    cfg = convgru_training_experiment.as_buildable()
    train_from_config(cfg)
    mock_build.assert_called_once()


@patch("mlcast.config.orchestrator.fdl.build")
def test_train_from_config_valid_latent_diffusion(mock_build: Any, tmp_path: Path) -> None:
    """Verify that a valid latent diffusion configuration passes validation and builds."""
    mock_build.return_value.trainer.log_dir = str(tmp_path)
    cfg = latent_diffusion_experiment.as_buildable()
    train_from_config(cfg)
    mock_build.assert_called_once()

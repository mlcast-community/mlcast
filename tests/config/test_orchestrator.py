from unittest.mock import patch

from mlcast.config import train_from_config, training_experiment


@patch("mlcast.config.orchestrator.fdl.build")
def test_train_from_config_valid(mock_build, tmp_path):
    """Verify that a valid configuration passes validation and builds."""
    mock_build.return_value.trainer.log_dir = str(tmp_path)
    cfg = training_experiment.as_buildable()
    train_from_config(cfg)
    mock_build.assert_called_once()

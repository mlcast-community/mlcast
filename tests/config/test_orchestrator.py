from unittest.mock import patch

from mlcast.config import train_from_config, training_experiment


@patch("mlcast.config.orchestrator.fdl.build")
def test_train_from_config_valid(mock_build, tmp_path):
    """Verify that a valid configuration passes validation and builds."""
    # default_root_dir is where _log_experiment_config_yaml_file falls back to
    # writing config.yaml when the (mocked) logger isn't a recognised type.
    # Point it at tmp_path so the write lands in pytest's tmp dir, not the repo.
    mock_build.return_value.trainer.default_root_dir = str(tmp_path)
    cfg = training_experiment.as_buildable()
    train_from_config(cfg)
    mock_build.assert_called_once()

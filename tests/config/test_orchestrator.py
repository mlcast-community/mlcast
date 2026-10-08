from unittest.mock import MagicMock, patch

from pytorch_lightning.loggers import MLFlowLogger

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


@patch("mlcast.config.orchestrator.fdl.build")
def test_train_from_config_routes_mlflow_logger_through_resilient_path(mock_build, tmp_path):
    """An MLFlowLogger-backed run logs hyperparams via mlcast.config.mlflow, not log_hyperparams."""
    mock_build.return_value.trainer.default_root_dir = str(tmp_path)
    mlflow_logger = MagicMock(spec=MLFlowLogger)
    mlflow_logger.run_id = "run-123"
    mlflow_logger.experiment = MagicMock()
    mock_build.return_value.trainer.logger = mlflow_logger

    cfg = training_experiment.as_buildable()
    train_from_config(cfg)

    mlflow_logger.experiment.log_batch.assert_called()
    mlflow_logger.log_hyperparams.assert_not_called()

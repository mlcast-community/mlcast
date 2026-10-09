from pytorch_lightning.loggers import MLFlowLogger

from mlcast.config import set_variables, toggle_masking, training_experiment, use_mlflow_logger


def test_fiddler_set_variables():
    """Verify set_variables syncs dataset variables and network input_channels."""
    cfg = training_experiment.as_buildable()

    # Apply fiddler
    set_variables(cfg, ["rainfall_rate", "rainfall_flux"])

    # Check sync
    assert cfg.data.dataset_factory.standard_names == ["rainfall_rate", "rainfall_flux"]
    assert cfg.pl_module.network.input_channels == 2


def test_fiddler_toggle_masking():
    """Verify toggle_masking syncs dataset mask return and module masked_loss."""
    cfg = training_experiment.as_buildable()

    # Disable masking
    toggle_masking(cfg, False)
    assert cfg.data.dataset_factory.return_mask is False
    assert cfg.pl_module.masked_loss is False

    # Enable masking
    toggle_masking(cfg, True)
    assert cfg.data.dataset_factory.return_mask is True
    assert cfg.pl_module.masked_loss is True


def test_fiddler_use_mlflow_logger_explicit_tracking_uri(monkeypatch):
    """An explicit tracking_uri is applied without needing MLFLOW_TRACKING_URI set."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg, tracking_uri="https://mlflow.example.org/")

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert cfg.trainer.logger.tracking_uri == "https://mlflow.example.org/"


def test_fiddler_use_mlflow_logger_defers_to_env_var(monkeypatch):
    """With no explicit tracking_uri, the logger config carries no tracking_uri of its own."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = training_experiment.as_buildable()

    use_mlflow_logger(cfg)

    assert cfg.trainer.logger.__fn_or_cls__ is MLFlowLogger
    assert not hasattr(cfg.trainer.logger, "tracking_uri") or cfg.trainer.logger.tracking_uri is None

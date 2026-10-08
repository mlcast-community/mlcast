from unittest.mock import MagicMock

from pytorch_lightning.loggers import MLFlowLogger

from mlcast.config.mlflow import log_hyperparams


def _mlflow_logger_mock(run_id="run-123"):
    logger = MagicMock(spec=MLFlowLogger)
    logger.run_id = run_id
    logger.experiment = MagicMock()
    return logger


def test_log_hyperparams_happy_path():
    logger = _mlflow_logger_mock()

    n_logged, n_failed = log_hyperparams(logger, {"a": 1, "b": "x", "c": 3.5})

    assert (n_logged, n_failed) == (3, 0)
    logger.experiment.log_batch.assert_called_once()
    _, kwargs = logger.experiment.log_batch.call_args
    assert len(kwargs["params"]) == 3


def test_log_hyperparams_batch_failure_falls_back_to_per_key(capsys):
    logger = _mlflow_logger_mock()

    def log_batch_side_effect(run_id, params):
        if len(params) > 1:
            raise RuntimeError("simulated batch failure")

    logger.experiment.log_batch.side_effect = log_batch_side_effect

    n_logged, n_failed = log_hyperparams(logger, {"a": 1, "b": 2, "c": 3})

    assert (n_logged, n_failed) == (3, 0)
    # 1 failed batch call + 3 successful per-key retries
    assert logger.experiment.log_batch.call_count == 4
    assert "retrying key by key" in capsys.readouterr().out


def test_log_hyperparams_skips_persistently_failing_key(capsys):
    logger = _mlflow_logger_mock()

    def log_batch_side_effect(run_id, params):
        if len(params) > 1:
            raise RuntimeError("simulated batch failure")
        if params[0].key == "bad_key":
            raise RuntimeError("simulated persistent failure")

    logger.experiment.log_batch.side_effect = log_batch_side_effect

    n_logged, n_failed = log_hyperparams(logger, {"good_key": 1, "bad_key": 2})

    assert (n_logged, n_failed) == (1, 1)
    out = capsys.readouterr().out
    assert "could not log MLflow hyperparameter 'bad_key'" in out


def test_log_hyperparams_truncates_long_keys_and_values():
    from mlflow.utils.validation import MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH

    logger = _mlflow_logger_mock()
    long_key = "k" * (MAX_ENTITY_KEY_LENGTH + 50)
    long_value = "v" * (MAX_PARAM_VAL_LENGTH + 1000)

    log_hyperparams(logger, {long_key: long_value})

    _, kwargs = logger.experiment.log_batch.call_args
    (param,) = kwargs["params"]
    assert len(param.key) == MAX_ENTITY_KEY_LENGTH
    assert len(param.value) == MAX_PARAM_VAL_LENGTH

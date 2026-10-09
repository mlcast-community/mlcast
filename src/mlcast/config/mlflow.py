"""Resilient MLflow logging helpers.

``pytorch_lightning.loggers.MLFlowLogger.log_hyperparams`` truncates
parameter *values* to MLflow's server-side limit but not *keys*, and has no
error handling: if the underlying ``log_batch`` call raises for any reason
(an overlong key, MLflow's immutable-param-rewrite rule, a transient server
error, an invalid character) the exception propagates straight out and can
kill a training run before it starts. :func:`log_hyperparams` below truncates
both keys and values up front, logs in batches, and falls back to
one-key-at-a-time logging on a batch failure, skipping (and warning about)
any key that still fails rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning.loggers import MLFlowLogger


def log_hyperparams(logger: MLFlowLogger, params: dict) -> tuple[int, int]:
    """Log flattened hyperparameters to MLflow, tolerating per-key failures.

    Parameters
    ----------
    logger : MLFlowLogger
        The active MLflow logger; ``logger.experiment`` is the underlying
        ``MlflowClient`` and ``logger.run_id`` identifies the run.
    params : dict
        Flattened hyperparameters; values may be of any type and are
        coerced to ``str`` here.

    Returns
    -------
    tuple[int, int]
        ``(n_logged, n_failed)`` counts.
    """
    from mlflow.entities import Param
    from mlflow.utils.validation import MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH, MAX_PARAMS_TAGS_PER_BATCH

    items = []
    for key, value in params.items():
        value = str(value)
        if len(value) > MAX_PARAM_VAL_LENGTH:
            value = value[: MAX_PARAM_VAL_LENGTH - 3] + "..."
        items.append(Param(key[:MAX_ENTITY_KEY_LENGTH], value))

    client = logger.experiment
    run_id = logger.run_id
    n_logged = n_failed = 0
    for i in range(0, len(items), MAX_PARAMS_TAGS_PER_BATCH):
        chunk = items[i : i + MAX_PARAMS_TAGS_PER_BATCH]
        try:
            client.log_batch(run_id, params=chunk)
            n_logged += len(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: MLflow hyperparameter batch failed ({e!r}); retrying key by key")
            for p in chunk:
                try:
                    client.log_batch(run_id, params=[p])
                    n_logged += 1
                except Exception as e2:  # noqa: BLE001
                    n_failed += 1
                    print(f"WARNING: could not log MLflow hyperparameter {p.key!r}: {e2!r}")

    print(f"Logged flattened Fiddle configuration to trainer.logger ({n_logged} logged, {n_failed} failed)")
    return n_logged, n_failed

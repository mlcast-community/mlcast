"""PyTorch Lightning callbacks for mlcast."""

import logging
import platform
import subprocess
import sys

import mlflow.system_metrics.metrics.gpu_monitor as _gpu_monitor_mod
import psutil
import pytorch_lightning as pl
import torch
from loguru import logger as log
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import MLFlowLogger

_NVML_UNSUPPORTED_GPU_PREFIXES = ("GB10", "Orin", "Xavier", "TX2", "Nano")


class _SuppressGpuWarningsFilter(logging.Filter):
    """Logging filter that silences WARNING-level records from the MLflow GPU monitor.

    Using a filter rather than setLevel because logging.config.dictConfig (called by
    mlflow.__init__) resets logger levels on existing loggers but does not clear their
    filters — so this approach is robust to any subsequent dictConfig calls.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def _suppress_gpu_monitor_warnings_if_needed() -> None:
    """Attach a filter to the MLflow GPU monitor logger for embedded NVIDIA GPUs.

    Embedded/Jetson GPUs (e.g. GB10) do not support all NVML queries. MLflow's
    gpu_monitor emits a WARNING on every polling interval for each unsupported query.
    We attach a filter (not setLevel) because dictConfig resets logger levels but
    preserves filters, making this robust to mlflow's internal logger reconfiguration.
    """
    if not torch.cuda.is_available():
        return
    if not any(
        torch.cuda.get_device_name(i).startswith(_NVML_UNSUPPORTED_GPU_PREFIXES)
        for i in range(torch.cuda.device_count())
    ):
        return
    _gpu_monitor_mod._logger.addFilter(_SuppressGpuWarningsFilter())


def _get_system_tags() -> dict[str, str]:
    """Collect system and environment metadata as a flat string dict."""
    tags = {
        "sys.hostname": platform.node(),
        "sys.os": platform.platform(),
        "sys.python_version": f"{platform.python_implementation()} {sys.version.split()[0]}",
        "sys.python_executable": sys.executable,
        "sys.command": " ".join(sys.argv),
        "sys.cpu_count": str(psutil.cpu_count(logical=False)),
        "sys.logical_cpu_count": str(psutil.cpu_count(logical=True)),
        "sys.gpu_count": str(torch.cuda.device_count()),
    }

    for i in range(torch.cuda.device_count()):
        tags[f"sys.gpu_{i}_type"] = torch.cuda.get_device_name(i)

    try:
        tags["sys.git_repository"] = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        tags["sys.git_commit"] = commit
        tags["sys.git_branch"] = branch
        tags["sys.git_state"] = f'git checkout -b "{branch}" {commit}'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return tags


class LogSystemInfoCallback(pl.Callback):
    """Logs system and environment metadata as MLflow run tags at training start.

    Collects hostname, OS, Python version, git state, CPU/GPU info and logs
    them as tags on the active MLflow run. Also starts MLflow's
    SystemMetricsMonitor to log CPU/GPU/memory metrics throughout training.
    No-ops silently for other loggers.
    """

    def __init__(self) -> None:
        self._system_monitor: SystemMetricsMonitor | None = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        tags = _get_system_tags()
        for key, value in tags.items():
            trainer.logger.experiment.set_tag(trainer.logger.run_id, key, value)

        tracking_uri = trainer.logger._tracking_uri
        experiment_id = trainer.logger._experiment_id
        run_id = trainer.logger.run_id

        self._system_monitor = SystemMetricsMonitor(run_id=run_id, tracking_uri=tracking_uri)
        _suppress_gpu_monitor_warnings_if_needed()
        self._system_monitor.start()

        run_url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        log.info(f"MLflow run started: {run_url}")

    def _stop_monitor(self) -> None:
        if self._system_monitor is not None:
            self._system_monitor.finish()
            self._system_monitor = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._stop_monitor()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self._stop_monitor()

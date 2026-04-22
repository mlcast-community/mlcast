"""PyTorch Lightning callbacks for mlcast."""

import platform
import subprocess
import sys

import psutil
import pytorch_lightning as pl
import torch
from loguru import logger as log
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning.loggers import MLFlowLogger


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

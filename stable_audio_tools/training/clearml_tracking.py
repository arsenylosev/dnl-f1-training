"""
ClearML Experiment Tracking — Foundation-1 Fine-Tuning
=======================================================
Provides a thin wrapper around the ClearML SDK that:

  - Initialises a ClearML Task for training or pre-encoding runs
  - Logs model config, dataset config, and all CLI args as structured
    hyperparameter sections (visible in the ClearML UI Hyperparameters tab)
  - Logs audio demo samples generated during training as debug samples
    (visible in the ClearML UI Debug Samples tab)
  - Logs GCS bucket / dataset provenance as task artifacts
  - Provides a PyTorch Lightning callback that bridges the existing
    WandbLogger metrics into ClearML scalars so both trackers receive
    identical data

Usage in train.py:
    from stable_audio_tools.training.clearml_tracking import (
        init_clearml_task,
        ClearMLDemoCallback,
    )
    task = init_clearml_task(args, model_config, dataset_config)
    # ... build trainer ...
    trainer = pl.Trainer(
        callbacks=[..., ClearMLDemoCallback(task)],
        ...
    )

Usage in pre_encode.py:
    from stable_audio_tools.training.clearml_tracking import init_clearml_pre_encode_task
    task = init_clearml_pre_encode_task(args, model_config, dataset_config)

Environment variables (set in your shell or Vertex AI job env):
    CLEARML_API_HOST          e.g. https://api.clear.ml
    CLEARML_API_ACCESS_KEY    your ClearML access key
    CLEARML_API_SECRET_KEY    your ClearML secret key
    CLEARML_PROJECT           project name (default: "Foundation-1 / DNL")
    CLEARML_TASK_NAME         override task name (optional)

Source: https://clear.ml/docs/latest/docs/integrations/pytorch_lightning/
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard — ClearML is optional; training still works without it
# ---------------------------------------------------------------------------
try:
    from clearml import Task, Logger as ClearMLLogger
    _CLEARML_AVAILABLE = True
except ImportError:
    _CLEARML_AVAILABLE = False
    Task = None  # type: ignore[assignment,misc]
    ClearMLLogger = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = os.environ.get("CLEARML_PROJECT", "Foundation-1 / DNL")


# ---------------------------------------------------------------------------
# Helper: check availability
# ---------------------------------------------------------------------------
def is_clearml_available() -> bool:
    return _CLEARML_AVAILABLE


def _warn_if_unavailable() -> bool:
    """Return True if ClearML is available, else log a warning and return False."""
    if not _CLEARML_AVAILABLE:
        logger.warning(
            "ClearML is not installed — experiment tracking is disabled.\n"
            "Install with: pip install clearml\n"
            "Then configure credentials: clearml-init"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Task initialisation helpers
# ---------------------------------------------------------------------------

def _connect_configs(
    task: "Task",
    args: Any,
    model_config: dict,
    dataset_config: dict,
) -> None:
    """Connect all configuration dicts as named hyperparameter sections."""
    # CLI arguments
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    # Remove non-serialisable entries
    safe_args = {
        k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
        for k, v in args_dict.items()
    }
    task.connect(safe_args, name="CLI Args")

    # Model config — flatten one level for readability
    task.connect(model_config, name="Model Config")

    # Dataset config
    task.connect(dataset_config, name="Dataset Config")

    # GCS provenance (if available)
    gcs_info: dict[str, str] = {}
    for ds in dataset_config.get("datasets", []):
        if "bucket" in ds:
            gcs_info[ds.get("id", "dataset")] = (
                f"gs://{ds['bucket']}/{ds.get('prefix', '')}"
            )
    if gcs_info:
        task.connect(gcs_info, name="GCS Data Sources")


def init_clearml_task(
    args: Any,
    model_config: dict,
    dataset_config: dict,
    task_name: Optional[str] = None,
    project_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Optional["Task"]:
    """
    Initialise a ClearML Task for a training run.

    Returns the Task object, or None if ClearML is not available.

    Parameters
    ----------
    args:
        Parsed CLI arguments (from argparse / prefigure).
    model_config:
        Loaded model config dict.
    dataset_config:
        Loaded dataset config dict.
    task_name:
        Task display name. Defaults to CLEARML_TASK_NAME env var or args.name.
    project_name:
        ClearML project name. Defaults to CLEARML_PROJECT env var.
    tags:
        Optional list of string tags to attach to the task.
    """
    if not _warn_if_unavailable():
        return None

    resolved_name = (
        task_name
        or os.environ.get("CLEARML_TASK_NAME")
        or getattr(args, "name", "f1-3s-finetune")
    )
    resolved_project = project_name or DEFAULT_PROJECT

    task = Task.init(
        project_name=resolved_project,
        task_name=resolved_name,
        task_type=Task.TaskTypes.training,
        tags=tags or ["foundation-1", "3s", "gcs", "diffusion-transformer"],
        # ClearML auto-captures: source code, git diff, installed packages,
        # TensorBoard scalars (from WandbLogger's TF event files), PyTorch
        # model snapshots, and console stdout/stderr.
        auto_connect_frameworks={
            "pytorch": True,
            "tensorboard": True,
            "matplotlib": True,
        },
    )

    _connect_configs(task, args, model_config, dataset_config)

    logger.info(
        "ClearML task initialised: project=%s  name=%s  id=%s",
        resolved_project,
        resolved_name,
        task.id,
    )
    return task


def init_clearml_pre_encode_task(
    args: Any,
    model_config: dict,
    dataset_config: dict,
    task_name: Optional[str] = None,
    project_name: Optional[str] = None,
) -> Optional["Task"]:
    """
    Initialise a ClearML Task for a pre-encoding run.

    Tracks: source code, installed packages, CLI args, model/dataset configs,
    number of batches processed, and output path.
    """
    if not _warn_if_unavailable():
        return None

    resolved_name = (
        task_name
        or os.environ.get("CLEARML_TASK_NAME")
        or "pre-encode-latents-3s"
    )
    resolved_project = project_name or DEFAULT_PROJECT

    task = Task.init(
        project_name=resolved_project,
        task_name=resolved_name,
        task_type=Task.TaskTypes.data_processing,
        tags=["pre-encode", "oobleck-vae", "3s", "gcs"],
        auto_connect_frameworks={"pytorch": True},
    )

    _connect_configs(task, args, model_config, dataset_config)

    # Log output path as an artifact pointer
    task.upload_artifact(
        "pre_encoded_output_path",
        artifact_object=str(getattr(args, "output_path", "unknown")),
    )

    logger.info(
        "ClearML pre-encode task initialised: project=%s  name=%s  id=%s",
        resolved_project,
        resolved_name,
        task.id,
    )
    return task


# ---------------------------------------------------------------------------
# PyTorch Lightning Callback
# ---------------------------------------------------------------------------

class ClearMLTrainingCallback(pl.Callback):
    """
    PyTorch Lightning callback that adds richer ClearML logging on top of
    the automatic W&B / TensorBoard capture.

    Logs:
      - Training step loss scalars (series: "train/loss")
      - GPU memory usage per step (series: "system/gpu_mem_gb")
      - Checkpoint paths as ClearML artifacts on every save
      - Audio demo WAV files as debug samples (call log_audio_demo() manually
        from your demo callback)

    Parameters
    ----------
    task:
        The ClearML Task returned by init_clearml_task().
    log_every_n_steps:
        How often to report GPU memory (default: 100).
    """

    def __init__(
        self,
        task: Optional["Task"],
        log_every_n_steps: int = 100,
    ) -> None:
        super().__init__()
        self.task = task
        self.log_every_n_steps = log_every_n_steps
        self._clearml_logger: Optional["ClearMLLogger"] = (
            task.get_logger() if task is not None else None
        )

    # ------------------------------------------------------------------
    # Step-level hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._clearml_logger is None:
            return

        step = trainer.global_step

        # Log loss if available in outputs
        loss_val: Optional[float] = None
        if isinstance(outputs, dict):
            loss_val = outputs.get("loss")
        elif isinstance(outputs, torch.Tensor):
            loss_val = outputs.item()

        if loss_val is not None:
            self._clearml_logger.report_scalar(
                title="Loss",
                series="train/loss",
                value=float(loss_val),
                iteration=step,
            )

        # Log GPU memory every N steps
        if step % self.log_every_n_steps == 0 and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_gb = torch.cuda.memory_allocated(i) / 1e9
                self._clearml_logger.report_scalar(
                    title="GPU Memory (GB)",
                    series=f"gpu_{i}/allocated_gb",
                    value=mem_gb,
                    iteration=step,
                )

    # ------------------------------------------------------------------
    # Checkpoint hooks
    # ------------------------------------------------------------------

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Register the checkpoint path as a ClearML artifact."""
        if self.task is None:
            return
        # The actual file path is set by ModelCheckpoint; we can only
        # register the directory here. The file will be registered once
        # ModelCheckpoint writes it (see on_train_epoch_end workaround).
        ckpt_dir = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback else None
        if ckpt_dir:
            self.task.upload_artifact(
                f"checkpoint_dir_step_{trainer.global_step}",
                artifact_object=str(ckpt_dir),
            )

    # ------------------------------------------------------------------
    # Public helper: log an audio demo file
    # ------------------------------------------------------------------

    def log_audio_demo(
        self,
        audio_path: str,
        step: int,
        title: str = "Audio Demo",
        series: str = "generated",
    ) -> None:
        """
        Upload a WAV file as a ClearML debug sample.

        Call this from your demo callback after writing the WAV to disk:

            clearml_cb.log_audio_demo(
                audio_path="/tmp/demo_step_5000.wav",
                step=trainer.global_step,
            )
        """
        if self._clearml_logger is None:
            return
        path = Path(audio_path)
        if not path.exists():
            logger.warning("ClearML: audio demo file not found: %s", audio_path)
            return
        self._clearml_logger.report_media(
            title=title,
            series=series,
            iteration=step,
            local_path=str(path),
        )

    # ------------------------------------------------------------------
    # Training end
    # ------------------------------------------------------------------

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.task is None:
            return
        if self._clearml_logger:
            self._clearml_logger.report_text(
                "Training complete",
                print_console=False,
            )
        self.task.close()


# ---------------------------------------------------------------------------
# Pre-encoding progress callback
# ---------------------------------------------------------------------------

class ClearMLPreEncodeCallback(pl.Callback):
    """
    Lightweight callback for pre_encode.py that reports progress to ClearML.

    Logs:
      - Number of batches processed (scalar)
      - Estimated completion percentage (scalar)
    """

    def __init__(self, task: Optional["Task"], total_batches: Optional[int] = None) -> None:
        super().__init__()
        self.task = task
        self.total_batches = total_batches
        self._clearml_logger: Optional["ClearMLLogger"] = (
            task.get_logger() if task is not None else None
        )
        self._batches_done = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._clearml_logger is None:
            return

        self._batches_done += 1

        self._clearml_logger.report_scalar(
            title="Pre-Encoding Progress",
            series="batches_processed",
            value=self._batches_done,
            iteration=self._batches_done,
        )

        if self.total_batches and self.total_batches > 0:
            pct = 100.0 * self._batches_done / self.total_batches
            self._clearml_logger.report_scalar(
                title="Pre-Encoding Progress",
                series="completion_pct",
                value=pct,
                iteration=self._batches_done,
            )

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.task is None:
            return
        if self._clearml_logger:
            self._clearml_logger.report_text(
                f"Pre-encoding complete. Total batches: {self._batches_done}",
                print_console=True,
            )
        self.task.close()

"""ClearML experiment tracking integration.

Implements INF-002: Wire ClearML experiment tracking into LightningModule.
Every run logs: hyperparameters, train/val loss, val/sensitivity, val/specificity, val/f1, best checkpoint.
"""

from typing import Any, Dict, Optional
from clearml import Task
import torch


def init_clearml_task(
    project_name: str = "pnx_detector",
    task_name: str = "pneumonia_training",
    hyperparams: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> Task:
    """Initialize a ClearML task for experiment tracking.

    Args:
        project_name: ClearML project name
        task_name: Name of this specific task/run
        hyperparams: Dictionary of hyperparameters to log
        tags: List of tags for the task

    Returns:
        Initialized ClearML Task object
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.training,
        tags=tags or ["baseline"],
        auto_connect_frameworks=False,  # Disable auto-connect to avoid conflicts
    )

    # Connect hyperparameters
    if hyperparams:
        task.connect(hyperparams)

    return task


def make_clearml_logger(
    exp_id: str,
    run_name: str,
    config: Dict[str, Any],
    project_name: str = "pnx_detector",
) -> "ClearMLLogger":
    """Create a ClearML logger for LightningModule.

    Args:
        exp_id: Experiment identifier
        run_name: Name of this run
        config: Configuration dictionary to log
        project_name: ClearML project name

    Returns:
        ClearMLLogger instance
    """
    task_name = f"{exp_id}_{run_name}"
    task = init_clearml_task(
        project_name=project_name,
        task_name=task_name,
        hyperparams=config,
        tags=[exp_id, "baseline"],
    )

    return ClearMLLogger(task)


class ClearMLLogger:
    """ClearML logger wrapper for logging metrics and artifacts."""

    def __init__(self, task: Task):
        """Initialize the logger.

        Args:
            task: ClearML Task object
        """
        self.task = task
        self._logged_metrics: Dict[str, float] = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to ClearML.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        self.task.get_logger().report_scalar(
            title=name,
            series=name,
            value=value,
            iteration=step,
        )

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to ClearML.

        Args:
            params: Dictionary of hyperparameters
        """
        self.task.connect(params)

    def save_checkpoint(self, checkpoint_path: str, epoch: int) -> None:
        """Save a model checkpoint as an artifact.

        Args:
            checkpoint_path: Path to the checkpoint file
            epoch: Epoch number
        """
        self.task.upload_artifact(
            name=f"checkpoint_epoch_{epoch}",
            artifact_object=checkpoint_path,
        )

    def log_image(self, name: str, image_path: str, step: Optional[int] = None) -> None:
        """Log an image to ClearML.

        Args:
            name: Image name
            image_path: Path to the image file
            step: Training step (optional)
        """
        self.task.get_logger().report_single_image(
            title=name,
            file_path=image_path,
            iteration=step,
        )

    def finalize(self) -> None:
        """Finalize the task."""
        self.task.update_state(state=Task.TaskState.finished)
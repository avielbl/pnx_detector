"""Experiment tracking module for pneumonia detection.

Implements INF-002: ClearML experiment tracking integration.
"""

from .clearml_logger import init_clearml_task, make_clearml_logger

__all__ = ["init_clearml_task", "make_clearml_logger"]
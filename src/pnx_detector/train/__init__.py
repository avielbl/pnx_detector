"""Training module for pneumonia detection."""

from .trainer import run_training_experiment, run_all_folds, main

__all__ = ["run_training_experiment", "run_all_folds", "main"]
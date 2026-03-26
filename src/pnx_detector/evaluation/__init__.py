"""Evaluation module for pneumonia detection.

Implements INF-004: Evaluation harness with test runner, confusion matrix,
per-class metrics (sensitivity, specificity, F1), and ROC-AUC.
"""

from .evaluator import EvaluationHarness

__all__ = ["EvaluationHarness"]
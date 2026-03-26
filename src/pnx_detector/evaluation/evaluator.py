"""Evaluation harness for pneumonia detection.

Implements INF-004: Evaluation harness with test runner, confusion matrix,
per-class metrics (sensitivity, specificity, F1), and ROC-AUC.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    accuracy: float
    sensitivity: float  # Recall for positive class (PNEUMONIA)
    specificity: float  # Recall for negative class (NORMAL)
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    precision_at_recall: Optional[float] = None  # Precision at 95% recall
    recall_at_precision: Optional[float] = None  # Recall at 95% precision


class EvaluationHarness:
    """Evaluation harness for pneumonia detection model.

    Implements INF-004: Test runner producing confusion matrix,
    per-class metrics (sensitivity, specificity, F1), ROC-AUC.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize the evaluation harness.

        Args:
            class_names: List of class names (default: ["NORMAL", "PNEUMONIA"])
        """
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.n_classes = len(self.class_names)

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Evaluate model predictions.

        Args:
            predictions: Array of predicted class labels
            labels: Array of true class labels
            probabilities: Array of class probabilities (for ROC-AUC)

        Returns:
            EvaluationMetrics object with all metrics
        """
        # Convert to numpy if torch tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        # Calculate metrics
        accuracy = self._calculate_accuracy(predictions, labels)
        sensitivity = self._calculate_sensitivity(predictions, labels)
        specificity = self._calculate_specificity(predictions, labels)
        precision = self._calculate_precision(predictions, labels)
        recall = sensitivity  # Same as sensitivity for binary
        f1_score = self._calculate_f1(predictions, labels)
        roc_auc = self._calculate_roc_auc(labels, probabilities)
        cm = self._calculate_confusion_matrix(predictions, labels)

        return EvaluationMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            confusion_matrix=cm,
        )

    def _calculate_accuracy(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate accuracy."""
        return np.mean(predictions == labels)

    def _calculate_sensitivity(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate sensitivity (recall for positive class)."""
        # For binary: sensitivity = TP / (TP + FN)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def _calculate_specificity(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate specificity (recall for negative class)."""
        # For binary: specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)

    def _calculate_precision(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate precision."""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def _calculate_f1(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate F1 score."""
        precision = self._calculate_precision(predictions, labels)
        recall = self._calculate_sensitivity(predictions, labels)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_roc_auc(
        self, labels: np.ndarray, probabilities: Optional[np.ndarray]
    ) -> float:
        """Calculate ROC-AUC score."""
        if probabilities is None:
            return 0.0

        if self.n_classes == 2:
            # Binary classification: use probability of positive class
            if len(probabilities.shape) == 1:
                return roc_auc_score(labels, probabilities)
            else:
                return roc_auc_score(labels, probabilities[:, 1])
        else:
            return roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")

    def _calculate_confusion_matrix(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> List[List[int]]:
        """Calculate confusion matrix."""
        return confusion_matrix(labels, predictions).tolist()

    def calculate_threshold_metrics(
        self,
        probabilities: np.ndarray,
        target_sensitivity: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate optimal threshold for target sensitivity.

        Args:
            probabilities: Array of class probabilities for positive class
            target_sensitivity: Target sensitivity (default: 0.95)

        Returns:
            Dictionary with threshold and resulting metrics
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(
            np.array([1 if y == 1 else 0 for y in np.array(probabilities > 0.5).flatten()]),
            probabilities.flatten() if len(probabilities.shape) == 1 else probabilities[:, 1],
        )

        # Find threshold that achieves target sensitivity
        target_idx = np.argmin(np.abs(tpr - target_sensitivity))
        optimal_threshold = thresholds[target_idx]

        # Calculate metrics at optimal threshold
        predictions = (probabilities.flatten() if len(probabilities.shape) == 1 else probabilities[:, 1]) >= optimal_threshold

        return {
            "optimal_threshold": float(optimal_threshold),
            "sensitivity": float(tpr[target_idx]),
            "specificity": float(1 - fpr[target_idx]),
            "precision": float(self._calculate_precision(predictions.astype(int), np.array([1 if y == 1 else 0 for y in np.array(probabilities > 0.5).flatten()]))),
        }

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and save confusion matrix.

        Args:
            predictions: Array of predicted class labels
            labels: Array of true class labels
            save_path: Path to save the figure (optional)
        """
        cm = confusion_matrix(labels, predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and save ROC curve.

        Args:
            labels: Array of true class labels
            probabilities: Array of class probabilities for positive class
            save_path: Path to save the figure (optional)
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        fpr, tpr, _ = roc_curve(
            np.array([1 if y == 1 else 0 for y in labels]),
            probabilities.flatten() if len(probabilities.shape) == 1 else probabilities[:, 1],
        )

        roc_auc = roc_auc_score(labels, probabilities)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> str:
        """Generate a text report of evaluation metrics.

        Args:
            predictions: Array of predicted class labels
            labels: Array of true class labels
            probabilities: Array of class probabilities (optional)

        Returns:
            Formatted text report
        """
        metrics = self.evaluate(predictions, labels, probabilities)
        report = [
            "=" * 50,
            "EVALUATION REPORT",
            "=" * 50,
            "",
            f"Accuracy:  {metrics.accuracy:.4f}",
            f"Sensitivity: {metrics.sensitivity:.4f}",
            f"Specificity: {metrics.specificity:.4f}",
            f"Precision: {metrics.precision:.4f}",
            f"Recall: {metrics.recall:.4f}",
            f"F1 Score: {metrics.f1_score:.4f}",
            f"ROC-AUC: {metrics.roc_auc:.4f}",
            "",
            "Confusion Matrix:",
        ]

        for i, class_name in enumerate(self.class_names):
            row = [f"{self.class_names[j]}: {metrics.confusion_matrix[i][j]}" for j in range(self.n_classes)]
            report.append(f"  {class_name}: {' '.join(row)}")

        report.append("")
        report.append("=" * 50)

        return "\n".join(report)
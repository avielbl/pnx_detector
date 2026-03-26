"""Threshold optimization module for pneumonia detection.

Implements INF-006: Threshold optimization module to minimize false positive rate
while maintaining >95% sensitivity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from sklearn.metrics import roc_curve, precision_recall_curve


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    metrics_curve: List[Dict[str, float]]


class ThresholdOptimizer:
    """Threshold optimization for pneumonia detection.

    Implements INF-006: Threshold tuning to minimize false positive rate
    while maintaining >95% sensitivity.
    """

    def __init__(
        self,
        target_sensitivity: float = 0.95,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the threshold optimizer.

        Args:
            target_sensitivity: Target sensitivity (default: 0.95)
            class_names: List of class names (default: ["NORMAL", "PNEUMONIA"])
        """
        self.target_sensitivity = target_sensitivity
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]

    def optimize(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        min_sensitivity: float = 0.95,
        max_sensitivity: float = 1.0,
    ) -> ThresholdResult:
        """Find optimal threshold for target sensitivity.

        Args:
            probabilities: Array of class probabilities for positive class
            labels: Array of true class labels
            min_sensitivity: Minimum acceptable sensitivity
            max_sensitivity: Maximum acceptable sensitivity

        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        # Convert to numpy if torch tensors
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Ensure probabilities are for positive class
        if len(probabilities.shape) > 1:
            probabilities = probabilities[:, 1]

        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probabilities)

        # Find thresholds that achieve target sensitivity range
        valid_indices = np.where(
            (tpr >= min_sensitivity) & (tpr <= max_sensitivity)
        )[0]

        if len(valid_indices) == 0:
            # Fallback: use threshold at target sensitivity
            target_idx = np.argmin(np.abs(tpr - self.target_sensitivity))
            optimal_threshold = thresholds[target_idx]
        else:
            # Find threshold with highest specificity (lowest FPR) in valid range
            best_idx = valid_indices[np.argmin(fpr[valid_indices])]
            optimal_threshold = thresholds[best_idx]

        # Calculate metrics at optimal threshold
        predictions = (probabilities >= optimal_threshold).astype(int)
        metrics = self._calculate_metrics(predictions, labels, probabilities)

        # Generate full metrics curve for documentation
        metrics_curve = self._generate_metrics_curve(probabilities, labels)

        return ThresholdResult(
            optimal_threshold=float(optimal_threshold),
            sensitivity=metrics["sensitivity"],
            specificity=metrics["specificity"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            false_positive_rate=metrics["fpr"],
            metrics_curve=metrics_curve,
        )

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate all metrics at a given threshold.

        Args:
            predictions: Binary predictions
            labels: True labels
            probabilities: Class probabilities

        Returns:
            Dictionary of metrics
        """
        # True positives, false positives, true negatives, false negatives
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 Score
        f1_score = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": sensitivity,
            "f1_score": f1_score,
            "fpr": fpr,
        }

    def _generate_metrics_curve(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_points: int = 100,
    ) -> List[Dict[str, float]]:
        """Generate metrics at different thresholds.

        Args:
            probabilities: Class probabilities
            labels: True labels
            num_points: Number of threshold points to evaluate

        Returns:
            List of metric dictionaries
        """
        # Generate thresholds
        thresholds = np.linspace(0, 1, num_points)

        metrics_curve = []
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            metrics = self._calculate_metrics(predictions, labels, probabilities)
            metrics["threshold"] = float(threshold)
            metrics_curve.append(metrics)

        return metrics_curve

    def plot_threshold_curve(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot sensitivity-specificity tradeoff curve.

        Args:
            probabilities: Class probabilities
            labels: True labels
            save_path: Path to save the figure (optional)
        """
        import matplotlib.pyplot as plt

        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        specificity = 1 - fpr

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(specificity, tpr, label="ROC Curve", color="blue")
        ax.set_xlabel("Specificity", fontsize=12)
        ax.set_ylabel("Sensitivity (Recall)", fontsize=12)
        ax.set_title("Sensitivity-Specificity Tradeoff", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark target sensitivity point
        target_idx = np.argmin(np.abs(tpr - self.target_sensitivity))
        ax.plot(
            specificity[target_idx],
            tpr[target_idx],
            "ro",
            markersize=10,
            label=f"Target (Sensitivity={self.target_sensitivity:.2f})",
        )

        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_tradeoff_report(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """Generate a text report of threshold tradeoffs.

        Args:
            probabilities: Class probabilities
            labels: True labels

        Returns:
            Formatted text report
        """
        # Generate metrics curve
        metrics_curve = self._generate_metrics_curve(probabilities, labels)

        # Find best threshold for target sensitivity
        result = self.optimize(probabilities, labels)

        report = [
            "=" * 70,
            "THRESHOLD OPTIMIZATION REPORT",
            "=" * 70,
            "",
            f"Target Sensitivity: {self.target_sensitivity:.2f}",
            "",
            "OPTIMAL THRESHOLD:",
            f"  Threshold: {result.optimal_threshold:.4f}",
            f"  Sensitivity: {result.sensitivity:.4f}",
            f"  Specificity: {result.specificity:.4f}",
            f"  Precision: {result.precision:.4f}",
            f"  F1 Score: {result.f1_score:.4f}",
            f"  False Positive Rate: {result.false_positive_rate:.4f}",
            "",
            "THRESHOLD TRADEOFF CURVE:",
            "-" * 70,
            f"{'Threshold':<12}{'Sensitivity':<14}{'Specificity':<14}{'FPR':<12}",
            "-" * 70,
        ]

        # Add sample points from the curve
        step = max(1, len(metrics_curve) // 10)
        for metrics in metrics_curve[::step]:
            report.append(
                f"{metrics['threshold']:<12.4f}"
                f"{metrics['sensitivity']:<14.4f}"
                f"{metrics['specificity']:<14.4f}"
                f"{metrics['fpr']:<12.4f}"
            )

        report.append("-" * 70)
        report.append("")
        report.append("=" * 70)

        return "\n".join(report)
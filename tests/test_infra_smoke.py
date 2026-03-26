"""Infrastructure smoke tests for pneumonia detection.

Tests all INF-* components with dummy data to verify infrastructure is ready.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os


# ============================================================================
# INF-001: Data Pipeline Smoke Tests
# ============================================================================

class TestDataPipeline:
    """Smoke tests for data pipeline (INF-001)."""

    def test_transforms_output_shape(self):
        """Test that transforms produce correct output shape."""
        from pnx_detector.data.transforms import get_train_transforms, get_val_transforms, TARGET_SIZE
        from PIL import Image

        train_transform = get_train_transforms()
        val_transform = get_val_transforms()

        # Create dummy PIL image
        dummy_image = Image.new('RGB', (224, 224), color='gray')

        # Test train transforms
        output = train_transform(dummy_image)
        assert output.shape == (3, TARGET_SIZE, TARGET_SIZE), "Train transform output shape mismatch"

        # Test val transforms
        output = val_transform(dummy_image)
        assert output.shape == (3, TARGET_SIZE, TARGET_SIZE), "Val transform output shape mismatch"

    def test_class_weights(self):
        """Test that class weights are correctly defined."""
        from pnx_detector.data.dataloader import CLASS_WEIGHTS, CLASS_NAMES

        assert len(CLASS_WEIGHTS) == 2, "Should have 2 class weights"
        assert CLASS_NAMES == ["NORMAL", "PNEUMONIA"], "Class names mismatch"
        assert CLASS_WEIGHTS[0] == 1.94, "NORMAL class weight should be 1.94"
        assert CLASS_WEIGHTS[1] == 0.67, "PNEUMONIA class weight should be 0.67"


# ============================================================================
# INF-002: Experiment Tracking Smoke Tests
# ============================================================================

class TestExperimentTracking:
    """Smoke tests for experiment tracking (INF-002)."""

    def test_clearml_logger_import(self):
        """Test that ClearML logger can be imported."""
        from pnx_detector.tracking import init_clearml_task, make_clearml_logger
        assert init_clearml_task is not None
        assert make_clearml_logger is not None

    def test_clearml_logger_class(self):
        """Test ClearML logger class structure."""
        from pnx_detector.tracking.clearml_logger import ClearMLLogger

        # Just verify the class exists and has expected methods
        assert hasattr(ClearMLLogger, "log_metric")
        assert hasattr(ClearMLLogger, "log_hyperparams")
        assert hasattr(ClearMLLogger, "save_checkpoint")
        assert hasattr(ClearMLLogger, "log_image")
        assert hasattr(ClearMLLogger, "finalize")


# ============================================================================
# INF-003: Model Smoke Tests
# ============================================================================

class TestModel:
    """Smoke tests for model architecture (INF-003)."""

    def test_model_forward_pass(self):
        """Test that model forward pass succeeds with dummy data."""
        from pnx_detector.models import PneumoniaModel

        # Create model with dummy class weights
        class_weights = torch.tensor([1.94, 0.67])
        model = PneumoniaModel(num_classes=2, class_weights=class_weights)
        model.eval()

        # Create dummy batch
        dummy_batch = torch.rand(4, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            logits = model(dummy_batch)

        # Verify output shape
        assert logits.shape == (4, 2), "Output shape should be (batch, num_classes)"

    def test_model_loss_computation(self):
        """Test that loss computes without NaN."""
        from pnx_detector.models import PneumoniaModel

        class_weights = torch.tensor([1.94, 0.67])
        model = PneumoniaModel(num_classes=2, class_weights=class_weights)
        model.eval()

        # Dummy batch
        dummy_batch = torch.rand(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4,))

        # Forward pass and loss
        logits = model(dummy_batch)
        loss = model.criterion(logits, labels)

        # Verify loss is valid
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_model_backward_pass(self):
        """Test that backward pass completes."""
        from pnx_detector.models import PneumoniaModel

        class_weights = torch.tensor([1.94, 0.67])
        model = PneumoniaModel(num_classes=2, class_weights=class_weights)
        model.train()

        # Dummy batch
        dummy_batch = torch.rand(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4,))

        # Forward and backward
        logits = model(dummy_batch)
        loss = model.criterion(logits, labels)
        loss.backward()

        # Verify gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Model should have gradients after backward pass"


# ============================================================================
# INF-004: Evaluation Harness Smoke Tests
# ============================================================================

class TestEvaluationHarness:
    """Smoke tests for evaluation harness (INF-004)."""

    def test_evaluate_dummy_predictions(self):
        """Test evaluator runs on dummy predictions without error."""
        from pnx_detector.evaluation import EvaluationHarness

        evaluator = EvaluationHarness()

        # Dummy predictions and labels
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        labels = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        probabilities = np.array([0.3, 0.8, 0.4, 0.2, 0.9, 0.6, 0.7, 0.3])

        # Evaluate
        metrics = evaluator.evaluate(predictions, labels, probabilities)

        # Verify metrics are computed
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.sensitivity <= 1
        assert 0 <= metrics.specificity <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.roc_auc <= 1

    def test_confusion_matrix_computation(self):
        """Test confusion matrix is computed correctly."""
        from pnx_detector.evaluation import EvaluationHarness

        evaluator = EvaluationHarness()

        predictions = np.array([0, 1, 1, 0])
        labels = np.array([0, 1, 0, 0])

        metrics = evaluator.evaluate(predictions, labels)

        # Verify confusion matrix shape
        assert len(metrics.confusion_matrix) == 2
        assert len(metrics.confusion_matrix[0]) == 2

    def test_generate_report(self):
        """Test report generation."""
        from pnx_detector.evaluation import EvaluationHarness

        evaluator = EvaluationHarness()

        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])

        report = evaluator.generate_report(predictions, labels)

        # Verify report contains expected content
        assert "EVALUATION REPORT" in report
        assert "Accuracy" in report
        assert "Sensitivity" in report
        assert "Specificity" in report


# ============================================================================
# INF-005: Inference Engine Smoke Tests
# ============================================================================

class TestInferenceEngine:
    """Smoke tests for inference engine (INF-005)."""

    def test_inference_result_dataclass(self):
        """Test InferenceResult dataclass structure."""
        from pnx_detector.inference.engine import InferenceResult

        result = InferenceResult(
            class_name="PNEUMONIA",
            confidence=0.85,
            class_probabilities={"NORMAL": 0.15, "PNEUMONIA": 0.85},
        )

        assert result.class_name == "PNEUMONIA"
        assert result.confidence == 0.85
        assert "PNEUMONIA" in result.class_probabilities

    def test_gradcam_generator_structure(self):
        """Test Grad-CAM generator class structure."""
        from pnx_detector.inference.gradcam import GradCAM, GradCAMGenerator

        # Verify classes exist
        assert GradCAM is not None
        assert GradCAMGenerator is not None

        # Verify GradCAM has expected methods
        assert hasattr(GradCAM, "_register_hooks")
        assert hasattr(GradCAM, "_save_activation")
        assert hasattr(GradCAM, "_save_gradient")
        assert hasattr(GradCAM, "__call__")


# ============================================================================
# INF-006: Threshold Optimization Smoke Tests
# ============================================================================

class TestThresholdOptimizer:
    """Smoke tests for threshold optimization (INF-006)."""

    def test_optimize_dummy_data(self):
        """Test threshold optimization on dummy data."""
        from pnx_detector.utils import ThresholdOptimizer

        optimizer = ThresholdOptimizer(target_sensitivity=0.95)

        # Dummy probabilities and labels
        probabilities = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.4, 0.9])
        labels = np.array([0, 0, 0, 1, 0, 1, 0, 1])

        # Optimize
        result = optimizer.optimize(probabilities, labels)

        # Verify result structure
        assert 0 <= result.optimal_threshold <= 1
        assert 0 <= result.sensitivity <= 1
        assert 0 <= result.specificity <= 1
        assert len(result.metrics_curve) > 0

    def test_generate_tradeoff_report(self):
        """Test tradeoff report generation."""
        from pnx_detector.utils import ThresholdOptimizer

        optimizer = ThresholdOptimizer(target_sensitivity=0.95)

        probabilities = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.4, 0.9])
        labels = np.array([0, 0, 0, 1, 0, 1, 0, 1])

        report = optimizer.generate_tradeoff_report(probabilities, labels)

        # Verify report contains expected content
        assert "THRESHOLD OPTIMIZATION REPORT" in report
        assert "Target Sensitivity" in report
        assert "OPTIMAL THRESHOLD" in report


# ============================================================================
# Integration Smoke Test
# ============================================================================

class TestIntegration:
    """Integration smoke tests for the complete pipeline."""

    def test_complete_pipeline_with_dummy_data(self):
        """Test complete pipeline with dummy data."""
        from PIL import Image
        # 1. Data pipeline
        from pnx_detector.data.transforms import get_train_transforms
        train_transform = get_train_transforms()
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        transformed = train_transform(dummy_image)
        assert transformed.shape == (3, 224, 224)

        # 2. Model
        from pnx_detector.models import PneumoniaModel
        model = PneumoniaModel(num_classes=2)
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch := torch.rand(1, 3, 224, 224))
        assert output.shape == (1, 2)

        # 3. Evaluation
        from pnx_detector.evaluation import EvaluationHarness
        evaluator = EvaluationHarness()
        metrics = evaluator.evaluate(
            predictions=np.array([0, 1]),
            labels=np.array([0, 1]),
            probabilities=np.array([[0.5, 0.5], [0.3, 0.7]]),
        )
        assert metrics.accuracy >= 0

        # 4. Threshold optimization
        from pnx_detector.utils import ThresholdOptimizer
        optimizer = ThresholdOptimizer()
        result = optimizer.optimize(
            probabilities=np.array([0.3, 0.7, 0.2, 0.8]),
            labels=np.array([0, 1, 0, 1]),
        )
        assert result.optimal_threshold >= 0

        # All components work together
        assert True
"""Inference module for pneumonia detection.

Implements INF-005: Inference engine with Grad-CAM visualization.
CPU-based model loading, GPU-based saliency map generation.
"""

from .engine import InferenceEngine
from .gradcam import GradCAMGenerator

__all__ = ["InferenceEngine", "GradCAMGenerator"]
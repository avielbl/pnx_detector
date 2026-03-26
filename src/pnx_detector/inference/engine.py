"""Inference engine for pneumonia detection.

Implements INF-005: Inference engine with Grad-CAM visualization.
CPU-based model loading, GPU-based saliency map generation.
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import io


@dataclass
class InferenceResult:
    """Result of model inference."""

    class_name: str
    confidence: float
    class_probabilities: Dict[str, float]
    grad_cam_path: Optional[str] = None  # Path to Grad-CAM heatmap image


class InferenceEngine:
    """Inference engine for pneumonia detection.

    Implements INF-005: CPU-based model inference with Grad-CAM visualization.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        class_names: Optional[list] = None,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (default: CPU)
            class_names: List of class names (default: ["NORMAL", "PNEUMONIA"])
        """
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.device = device or torch.device("cpu")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> nn.Module:
        """Load a trained model from checkpoint.

        Args:
            model_path: Path to the model checkpoint

        Returns:
            Loaded model in eval mode
        """
        try:
            from pnx_detector.models import PneumoniaModel
        except ImportError:
            raise ImportError(
                "Please ensure pnx_detector is installed: uv run pip install -e ."
            )

        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Create model and load state dict
        if "state_dict" in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint["state_dict"]
            # Remove "model." prefix if present
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        model = PneumoniaModel(num_classes=2)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model

    def preprocess_image(
        self,
        image: Union[str, Image.Image],
        target_size: int = 224,
    ) -> torch.Tensor:
        """Preprocess an image for inference.

        Args:
            image: Path to image file or PIL Image
            target_size: Target image size

        Returns:
            Preprocessed tensor of shape (1, 3, target_size, target_size)
        """
        from pnx_detector.data.transforms import IMAGENET_MEAN, IMAGENET_STD

        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Apply preprocessing transforms
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        tensor = preprocess(image).unsqueeze(0)
        return tensor.to(self.device)

    def infer(
        self,
        image: Union[str, Image.Image],
        generate_grad_cam: bool = False,
        grad_cam_device: Optional[torch.device] = None,
    ) -> InferenceResult:
        """Run inference on a single image.

        Args:
            image: Path to image file or PIL Image
            generate_grad_cam: Whether to generate Grad-CAM heatmap
            grad_cam_device: Device for Grad-CAM (default: GPU if available)

        Returns:
            InferenceResult with predictions and optional Grad-CAM
        """
        # Preprocess image
        tensor = self.preprocess_image(image)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Get predictions
        confidence = probs[0, preds[0]].item()
        class_name = self.class_names[preds[0].item()]

        # Get class probabilities
        class_probabilities = {
            name: prob.item() for name, prob in zip(self.class_names, probs[0])
        }

        # Generate Grad-CAM if requested
        grad_cam_path = None
        if generate_grad_cam:
            grad_cam_device = grad_cam_device or (
                torch.device("cuda") if torch.cuda.is_available() else self.device
            )
            grad_cam_path = self._generate_grad_cam(image, grad_cam_device)

        return InferenceResult(
            class_name=class_name,
            confidence=confidence,
            class_probabilities=class_probabilities,
            grad_cam_path=grad_cam_path,
        )

    def _generate_grad_cam(
        self,
        image: Union[str, Image.Image],
        device: torch.device,
    ) -> str:
        """Generate Grad-CAM heatmap for the predicted class.

        Args:
            image: Path to image file or PIL Image
            device: Device to run Grad-CAM on

        Returns:
            Path to saved heatmap image
        """
        from pnx_detector.inference.gradcam import GradCAMGenerator

        # Generate Grad-CAM
        generator = GradCAMGenerator(model=self.model, device=device)
        heatmap_path = generator.generate(image)

        return heatmap_path

    def batch_infer(
        self,
        images: list,
        generate_grad_cam: bool = False,
    ) -> list:
        """Run inference on a batch of images.

        Args:
            images: List of image paths or PIL Images
            generate_grad_cam: Whether to generate Grad-CAM for each image

        Returns:
            List of InferenceResult objects
        """
        results = []
        for image in images:
            result = self.infer(image, generate_grad_cam=generate_grad_cam)
            results.append(result)
        return results

    def get_feature_vector(
        self,
        image: Union[str, Image.Image],
    ) -> torch.Tensor:
        """Extract feature vector from image.

        Args:
            image: Path to image file or PIL Image

        Returns:
            Feature vector tensor
        """
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            features = self.model.backbone(tensor)

        return features
"""Grad-CAM generator for pneumonia detection.

Implements INF-005: GPU-based saliency map generation for visual explanations.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Union
from pathlib import Path
import os


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model: nn.Module, target_layer: str):
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Name of the layer to use for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks for target layer."""
        target_module = self._get_target_module(self.model, self.target_layer)

        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        # Forward hook to capture activations
        target_module.register_forward_hook(self._save_activation)

        # Backward hook to capture gradients
        target_module.register_backward_hook(self._save_gradient)

    def _get_target_module(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get the target module by name.

        Args:
            model: PyTorch model
            layer_name: Name of the layer

        Returns:
            Target module or None if not found
        """
        # Handle different model structures
        if hasattr(model, "backbone"):
            model = model.backbone

        # Navigate to the layer
        parts = layer_name.split(".")
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """Save activation for backward pass."""
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        """Save gradient for backward pass."""
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            x: Input tensor of shape (batch, channels, height, width)
            target_class: Class to visualize (None = predicted class)

        Returns:
            Heatmap as numpy array
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Forward pass
        features = self.model.backbone(x)
        logits = self.model.classifier(features)

        # If target_class is None, use predicted class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)[0].item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        logits[:, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients or activations")

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))  # (batch, num_channels)

        # Weighted sum of activations
        cam = torch.sum(weights[:, :, None, None] * activations, dim=1)  # (batch, H, W)

        # ReLU (only positive contributions)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0].cpu().numpy()  # Return first batch item


class GradCAMGenerator:
    """Generator for Grad-CAM heatmaps."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        target_layer: str = "features.5.conv.3",
    ):
        """Initialize the Grad-CAM generator.

        Args:
            model: Trained PyTorch model
            device: Device to run Grad-CAM on
            target_layer: Target layer for Grad-CAM (EfficientNet-B4 conv layer)
        """
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.target_layer = target_layer
        self.gradcam = GradCAM(self.model, target_layer)

    def generate(
        self,
        image: Union[str, Image.Image],
        output_path: Optional[str] = None,
        target_class: Optional[int] = None,
    ) -> str:
        """Generate Grad-CAM heatmap for an image.

        Args:
            image: Path to image file or PIL Image
            output_path: Path to save the heatmap (optional)
            target_class: Class to visualize (None = predicted class)

        Returns:
            Path to saved heatmap image
        """
        from pnx_detector.data.transforms import IMAGENET_MEAN, IMAGENET_STD
        from torchvision import transforms

        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        tensor = preprocess(image).unsqueeze(0).to(self.device)

        # Generate Grad-CAM
        heatmap = self.gradcam(tensor, target_class=target_class)

        # Create output directory if needed
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save heatmap
        return self._save_heatmap(heatmap, image, output_path)

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        output_path: Optional[str] = None,
    ) -> str:
        """Save Grad-CAM heatmap overlaid on original image.

        Args:
            heatmap: Heatmap array
            original_image: Original PIL Image
            output_path: Path to save the result

        Returns:
            Path to saved image
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Resize heatmap to match image size
        heatmap_resized = plt.figure()
        plt.imshow(heatmap, cmap="jet", alpha=0.6)
        plt.axis("off")
        plt.tight_layout()

        # Get heatmap image
        heatmap_img = np.array(heatmap_resized.canvas.tostring_rgb())
        heatmap_resized.clf()
        plt.close()

        # Resize to match original image
        heatmap_img = Image.fromarray(heatmap_img).resize(original_image.size)

        # Overlay heatmap on original image
        overlay = Image.blend(original_image, heatmap_img, alpha=0.5)

        # Save
        if output_path is None:
            output_path = "/tmp/gradcam_output.png"

        overlay.save(output_path)

        return output_path

    def generate_batch(
        self,
        images: list,
        output_dir: Optional[str] = None,
    ) -> list:
        """Generate Grad-CAM heatmaps for a batch of images.

        Args:
            images: List of image paths or PIL Images
            output_dir: Directory to save heatmaps

        Returns:
            List of output paths
        """
        paths = []
        for i, image in enumerate(images):
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(output_dir, f"gradcam_{i}.png")
            else:
                output_path = None

            path = self.generate(image, output_path=output_path)
            paths.append(path)

        return paths
"""Inference demo: model prediction + Grad-CAM visualization on sample images.

Produces overlay images showing:
- Predicted class + confidence score
- Grad-CAM heatmap highlighting suspicious regions
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parents[3]))
from src.pnx_detector.models.efficientnet import PneumoniaModel
from src.pnx_detector.data.transforms import IMAGENET_MEAN, IMAGENET_STD

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GradCAMRunner:
    """Minimal, correct Grad-CAM for EfficientNet-B4 backbone."""

    def __init__(self, model: PneumoniaModel, device: torch.device):
        self.model = model
        self.device = device
        self._activations = None
        self._gradients = None

        # Hook onto the last conv block of EfficientNet-B4
        # timm EfficientNet: backbone.blocks[-1][-1].conv_pwl
        target = model.backbone.blocks[-1][-1].conv_pwl
        target.register_forward_hook(self._fwd_hook)
        target.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def __call__(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns Grad-CAM heatmap [0,1] for class_idx."""
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[:, class_idx].backward()

        grads = self._gradients          # (1, C, H, W)
        acts  = self._activations        # (1, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)     # (H, W)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).cpu().detach().numpy()
        return cam


def render_overlay(image_path: str, model: PneumoniaModel,
                   gradcam: GradCAMRunner, device: torch.device,
                   threshold: float = 0.50) -> tuple:
    """Run inference + Grad-CAM on one image. Returns (fig, result_dict)."""
    pil = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(pil).unsqueeze(0).to(device)

    # Forward pass for probability
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    prob_pnx = probs[1].item()
    pred_class = 1 if prob_pnx >= threshold else 0
    pred_name  = CLASS_NAMES[pred_class]
    confidence = probs[pred_class].item()

    # Grad-CAM for predicted class (requires grad)
    tensor_g = PREPROCESS(pil).unsqueeze(0).to(device).requires_grad_(True)
    heatmap = gradcam(tensor_g, pred_class)

    # Resize heatmap to original image dimensions
    h, w = pil.height, pil.width
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heatmap_np  = np.array(heatmap_pil) / 255.0

    # Build figure: original | overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pil, cmap="gray")
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(pil, cmap="gray")
    colored = cm.jet(heatmap_np)[:, :, :3]   # RGB, no alpha
    axes[1].imshow(colored, alpha=0.45)
    color  = "red" if pred_class == 1 else "green"
    title  = (f"Pred: {pred_name}  ({confidence*100:.1f}%)\n"
              f"P(PNEUMONIA)={prob_pnx*100:.1f}%  thr={threshold:.2f}")
    axes[1].set_title(title, fontsize=11, color=color, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout()

    result = {
        "image": image_path,
        "pred_class": pred_name,
        "confidence": confidence,
        "prob_pneumonia": prob_pnx,
        "threshold": threshold,
    }
    return fig, result


def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=
        "checkpoints/EXP-001/baseline/best_epoch=31_val/f1=0.9436.ckpt")
    parser.add_argument("--data-dir",   default="chest_xray_data")
    parser.add_argument("--output-dir", default="docs/experiments/inference_samples")
    parser.add_argument("--threshold",  type=float, default=0.50)
    parser.add_argument("--n-samples",  type=int,   default=6,
                        help="Images per class to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PneumoniaModel.load_from_checkpoint(
        args.checkpoint, map_location=device, strict=False)
    model.eval()
    model.to(device)
    print(f"Loaded: {args.checkpoint}")

    gradcam = GradCAMRunner(model, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect test images
    random.seed(42)
    test_images = {"NORMAL": [], "PNEUMONIA": []}
    for cls in CLASS_NAMES:
        cls_dir = Path(args.data_dir) / "test" / cls
        imgs = list(cls_dir.glob("*.jpeg"))
        random.shuffle(imgs)
        test_images[cls] = imgs[:args.n_samples]

    print(f"\nGenerating {args.n_samples} samples per class -> {out_dir}/")
    all_results = []

    for cls in CLASS_NAMES:
        for i, img_path in enumerate(test_images[cls]):
            fig, result = render_overlay(
                str(img_path), model, gradcam, device, threshold=args.threshold)
            out_path = out_dir / f"{cls.lower()}_{i+1:02d}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            all_results.append(result)
            correct_lbl = 1 if cls == "PNEUMONIA" else 0
            status = "OK" if result["pred_class"] == cls else "WRONG"
            print(f"  [{status}] {cls} sample {i+1}: pred={result['pred_class']} "
                  f"({result['confidence']*100:.1f}%)  -> {out_path.name}")

    # Summary
    correct  = sum(r["pred_class"] == ("PNEUMONIA" if "PNEUMONIA" in r["image"] else "NORMAL")
                   for r in all_results)
    print(f"\nSample accuracy: {correct}/{len(all_results)}")
    print(f"Output images:   {out_dir}/")


if __name__ == "__main__":
    main()

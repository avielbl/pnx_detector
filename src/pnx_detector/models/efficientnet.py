"""EfficientNet-B4 model for pneumonia detection.

Implements INF-003: LightningModule scaffold for EfficientNet-B4 architecture
with custom classification head and dropout for regularization.
"""

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Optional


class PneumoniaModel(LightningModule):
    """LightningModule for pneumonia detection using EfficientNet-B4.

    Implements INF-003: EfficientNet-B4 with custom classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        dropout: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        num_frozen_layers: int = 0,
    ):
        """Initialize the model.

        Args:
            num_classes: Number of output classes (2 for binary classification)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            dropout: Dropout rate for the classification head
            class_weights: Weights for handling class imbalance
            num_frozen_layers: Number of layers to freeze from backbone (0 = train from scratch)
        """
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.class_weights = class_weights
        self.num_frozen_layers = num_frozen_layers

        # Load EfficientNet-B4
        self._load_backbone()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1792, num_classes),
        )

        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)

        # For sensitivity/specificity calculation
        self.val_predictions = []
        self.val_labels = []

        # For test metrics
        self.test_predictions = []
        self.test_labels = []

        self.save_hyperparameters(ignore=["class_weights"])

    def _load_backbone(self) -> None:
        """Load EfficientNet-B4 backbone."""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm library is required for EfficientNet. "
                "Install with: uv add timm"
            )

        # Load EfficientNet-B4 trained from scratch (not pretrained)
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0,  # Remove default classifier for feature extraction
            drop_rate=0.2,
        )

        # Freeze layers if specified
        if self.num_frozen_layers > 0:
            self._freeze_layers(self.num_frozen_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze specified number of layers from the backbone.

        Args:
            num_layers: Number of layers to freeze
        """
        for param in list(self.backbone.parameters())[:-num_layers]:
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        # Log training metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_accuracy(preds, labels), on_step=True, on_epoch=True, prog_bar=False)

        if torch.cuda.is_available():
            self.log("gpu/memory_allocated_gb", torch.cuda.memory_allocated() / 1e9, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """Validation step."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Log validation metrics (pass probs to AUROC; others accept logits/preds)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_accuracy(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", self.val_recall(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_f1(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc(probs, labels), on_step=False, on_epoch=True, prog_bar=False)

        self.val_predictions.append(preds)
        self.val_labels.append(labels)

    def on_validation_epoch_end(self) -> None:
        """Compute sensitivity and specificity at epoch end."""
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.val_predictions)
        all_labels = torch.cat(self.val_labels)

        # Calculate sensitivity (recall for PNEUMONIA class=1) and specificity (recall for NORMAL class=0)
        # Use fresh per-class Recall metric to avoid double-accumulation into val_recall state
        if self.num_classes == 2:
            per_class_recall = Recall(task="multiclass", num_classes=2, average=None)(all_preds, all_labels)
            sensitivity = per_class_recall[1]  # PNEUMONIA recall
            specificity = per_class_recall[0]  # NORMAL recall

            self.log("val/sensitivity", sensitivity, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/specificity", specificity, on_step=False, on_epoch=True, prog_bar=True)

        # Clear buffers
        self.val_predictions.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx: int) -> None:
        """Test step."""
        images, labels = batch
        logits = self(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.test_predictions.append(preds)
        self.test_labels.append(labels)

    def on_test_epoch_end(self) -> None:
        """Compute metrics at test epoch end."""
        all_preds = torch.cat(self.test_predictions)
        all_labels = torch.cat(self.test_labels)

        accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)(all_preds, all_labels)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="micro")(all_preds, all_labels)
        auroc = AUROC(task="multiclass", num_classes=self.num_classes)(all_preds, all_labels)

        self.log("test/accuracy", accuracy, prog_bar=True)
        self.log("test/f1", f1, prog_bar=True)
        self.log("test/auroc", auroc, prog_bar=True)

        # Clear buffers
        self.test_predictions.clear()
        self.test_labels.clear()

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
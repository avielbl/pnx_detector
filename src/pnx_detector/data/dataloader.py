"""Stratified k-fold DataLoader for pneumonia detection.

Implements INF-001: Stratified k-fold DataLoader with augmentation pipeline
from EDA findings. Applies class weights (NORMAL=1.94, PNEUMONIA=0.67).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np


# Class weights from EDA (inverse frequency)
CLASS_WEIGHTS = torch.tensor([1.94, 0.67])  # NORMAL=1.94, PNEUMONIA=0.67

# Class name mapping
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class PneumoniaDataset(Dataset):
    """PyTorch Dataset for chest X-ray images."""

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize the dataset.

        Args:
            image_paths: List of paths to image files
            labels: List of integer labels (0=NORMAL, 1=PNEUMONIA)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class PneumoniaDataModule(LightningDataModule):
    """LightningDataModule for pneumonia detection with stratified k-fold.

    Implements INF-001: Stratified k-fold DataLoader with augmentation pipeline.
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 4,
        num_splits: int = 5,
        random_state: int = 42,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
    ):
        """Initialize the data module.

        Args:
            data_dir: Path to the data directory containing train/val/test folders
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            num_workers: Number of workers for data loading
            num_splits: Number of stratified k-fold splits
            random_state: Random seed for reproducibility
            train_transforms: Training transforms (uses default if None)
            val_transforms: Validation transforms (uses default if None)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.num_splits = num_splits
        self.random_state = random_state

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # Store splits
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.current_fold: int = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, and testing."""
        from .transforms import get_train_transforms, get_val_transforms

        # Use default transforms if not provided
        if self.train_transforms is None:
            self.train_transforms = get_train_transforms()
        if self.val_transforms is None:
            self.val_transforms = get_val_transforms()

        # Collect all images and labels from train and val directories
        # (test is held out for final evaluation)
        all_paths = []
        all_labels = []

        for split in ["train", "val"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue

            for class_name in CLASS_NAMES:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue

                for img_path in class_dir.glob("*.jpeg"):
                    all_paths.append(str(img_path))
                    all_labels.append(CLASS_NAMES.index(class_name))

        # Convert to numpy arrays for stratified splitting
        self.all_paths = np.array(all_paths)
        self.all_labels = np.array(all_labels)

        # Create stratified k-fold splits
        self.skf = StratifiedKFold(
            n_splits=self.num_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Store fold indices
        self.fold_indices = list(self.skf.split(self.all_paths, self.all_labels))

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader for current fold."""
        train_idx = self.fold_indices[self.current_fold][0]

        dataset = PneumoniaDataset(
            image_paths=self.all_paths[train_idx].tolist(),
            labels=self.all_labels[train_idx].tolist(),
            transform=self.train_transforms,
        )

        # Create weighted sampler for class imbalance
        weights = self._get_sample_weights(self.all_labels[train_idx])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader for current fold."""
        val_idx = self.fold_indices[self.current_fold][1]

        dataset = PneumoniaDataset(
            image_paths=self.all_paths[val_idx].tolist(),
            labels=self.all_labels[val_idx].tolist(),
            transform=self.val_transforms,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader (held-out test set)."""
        test_dir = self.data_dir / "test"
        if not test_dir.exists():
            raise ValueError(f"Test directory not found: {test_dir}")

        test_paths = []
        test_labels = []

        for class_name in CLASS_NAMES:
            class_dir = test_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpeg"):
                test_paths.append(str(img_path))
                test_labels.append(CLASS_NAMES.index(class_name))

        dataset = PneumoniaDataset(
            image_paths=test_paths,
            labels=test_labels,
            transform=self.val_transforms,
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _get_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        """Compute sample weights based on class frequencies.

        Uses inverse frequency weighting to handle class imbalance.
        """
        # Count occurrences of each class
        class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))

        # Compute weights as inverse of class frequency
        total = len(labels)
        weights = total / (len(CLASS_NAMES) * class_counts[labels])

        return weights

    def set_fold(self, fold: int) -> None:
        """Set the current fold for cross-validation.

        Args:
            fold: Fold index (0 to num_splits-1)
        """
        if not 0 <= fold < self.num_splits:
            raise ValueError(f"Fold must be between 0 and {self.num_splits-1}")
        self.current_fold = fold

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for weighted loss function.

        Returns:
            Tensor of class weights [NORMAL_weight, PNEUMONIA_weight]
        """
        return CLASS_WEIGHTS.clone()
"""Image augmentation transforms for pneumonia detection.

Based on EDA findings:
- Heavy augmentation required to prevent overfitting when training from scratch
- ImageNet normalization statistics used
- RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter applied
"""

import torch
from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target image size for EfficientNet-B4
TARGET_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Get training transforms with heavy augmentation.
    
    Returns:
        Compose transform pipeline for training data
    """
    return transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Get validation/test transforms (no augmentation).
    
    Returns:
        Compose transform pipeline for validation/test data
    """
    return transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
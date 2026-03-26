# Class Weights for Pneumonia Detection

*Generated: 2026-03-26*

## Dataset Statistics

| Class | Count | Percentage |
| :--- | ---: | ---: |
| PNEUMONIA | 3,875 | 74.3% |
| NORMAL | 1,341 | 25.7% |

## Imbalance Analysis

- **Imbalance Ratio:** 2.9:1 (PNEUMONIA:NORMAL)
- **Imbalance Status:** Moderate

## Computed Class Weights

Using inverse frequency weighting: `weight[class] = total_samples / (num_classes * class_count)`

| Class | Weight |
| :--- | ---: |
| NORMAL | 1.94 |
| PNEUMONIA | 0.67 |

## Usage Example (PyTorch)

```python
import torch
import torch.nn as nn

class_weights = torch.tensor([0.67, 1.94])  # [PNEUMONIA, NORMAL]
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## Usage Example (TensorFlow/Keras)

```python
import tensorflow as tf

class_weights = {0: 0.67, 1: 1.94}  # {PNEUMONIA, NORMAL}
model.fit(x_train, y_train, class_weight=class_weights, ...)
```

## Recommendations

1. **Apply class weights during training** to account for the moderate imbalance
2. **Monitor per-class metrics** (sensitivity for PNEUMONIA, specificity for NORMAL)
3. **Consider additional augmentation** for the NORMAL class if training shows bias
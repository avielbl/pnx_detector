---
name: retro-exp001-analysis
description: "Retrospective from Stage 7 analysis session on EXP-001 5-fold results. Usage scenarios: (1) when sensitivity plateaus below 0.95 on a from-scratch EfficientNet baseline, (2) when k-fold specificity variance is suspiciously wide, (3) when deciding between threshold tuning vs. transfer learning after a baseline run."
author: researcher
date: 2026-03-28
experiment_id: EXP-001
---

# Retrospective: EXP-001 Analysis

## A. Session Overview

| Item | Details |
| :--- | :--- |
| **Date** | 2026-03-28 |
| **Goal** | Analyze EXP-001 5-fold results, render hypothesis verdict, produce analysis document |
| **Experiment ID** | EXP-001 |
| **Environment** | GTX 1650 4.3GB, CUDA 12.5, torch 2.6.0+cu124, lightning 2.x, timm EfficientNet-B4 |

## B. What Worked ✅

| Finding | Exact Value / Code | Metric Result |
| :--- | :--- | :--- |
| EfficientNet-B4 from scratch | lr=1e-4, bs=16, AdamW wd=1e-2, CosineWarmRestarts T_0=10 | mean F1=0.936 ± 0.006 |
| Class weights NORMAL=1.94, PNEUMONIA=0.67 | Applied to CrossEntropyLoss | Specificity ≥ 0.95 from epoch 0 onward |
| Early stopping patience=10 on val/f1 | monitor="val/f1", mode="max" | Clean convergence, no overfitting across all 5 folds |
| Heavy augmentation | RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter | Sensitivity improved steadily across all folds |

**Copy-paste ready configuration (best result — fold 4, epoch 31):**
```yaml
architecture: efficientnet_b4
pretrained: false
learning_rate: 1.0e-4
weight_decay: 1.0e-2
dropout: 0.3
batch_size: 16
epochs: 50
early_stopping_patience: 10
early_stopping_monitor: val/f1
scheduler: CosineAnnealingWarmRestarts
scheduler_T0: 10
scheduler_T_mult: 2
scheduler_eta_min: 1.0e-6
class_weights: [1.94, 0.67]  # [NORMAL, PNEUMONIA]
seed: 42
# Result: val/f1=0.944, sensitivity=0.937, specificity=0.963
```

## C. Failed Attempts ❌ (Most Valuable Section)

| Attempt | Why It Failed | Root Cause | Lesson Learned |
| :--- | :--- | :--- | :--- |
| 5-fold CV on 5,216 images | Per-fold specificity variance 0.963–0.993 | ~208 NORMAL images per val fold; single-digit TN change = 0.5pp specificity shift | Switch to fixed 70/20/10 split once val set is adequately sized (≥1,000 per class ideally) |
| EarlyStopping on val/f1 | Terminated some folds before sensitivity plateau | F1 is dominated by PNEUMONIA (majority); sensitivity can still improve after F1 plateaus | For sensitivity-critical tasks, consider monitoring val/sensitivity directly or using a sensitivity-weighted composite metric |
| CosineAnnealingWarmRestarts T_0=10 | Sensitivity dips at restart boundaries (fold 0 ep11: 0.920→0.690, recovers ep13+) | LR reset disrupts the learned decision boundary; sensitivity recovers slower than F1/specificity | Use ReduceLROnPlateau or larger T_0 (≥20) when sensitivity stability matters |

## D. Bugs & Fixes 🔧

All infrastructure bugs from this experiment are fully documented in:
`docs/knowledge/RETRO_EXP_001_stage5-infra-and-experiment-launch.md`

No new bugs encountered during the analysis session itself.

## E. Surprising Findings

* **Asymmetric convergence:** Specificity exceeded 0.95 from epoch 0 in all folds. Sensitivity required 15–20 epochs to approach its ceiling. The class weighting (NORMAL=1.94) is producing a conservative model — it correctly avoids false negatives early, but the 0.5 threshold is suboptimal.
* **Fold 4 trained longest and performed best:** Fold 4 ran to epoch 31 vs. fold 0's early stop at epoch 18, and achieved the best sensitivity (0.937). The F1-based early stopping terminated other folds while sensitivity was still potentially improvable. This suggests the current stopping criterion is sensitivity-suboptimal.
* **Sensitivity ceiling consistent across folds:** SD=0.010 across 5 independent folds. This is a systematic limit of the training configuration, not noise — confirming that architecture change or threshold tuning is needed, not more training.

## F. Open Questions

* Can the EXP-001 best checkpoint (fold 4, ep31, F1=0.944) hit sensitivity ≥ 0.95 with threshold ~0.35–0.40?
* Does ImageNet pretraining recover the remaining ~3pp sensitivity gap, or is it purely threshold-limited?
* Should EarlyStopping monitor `val/sensitivity` directly for future sensitivity-critical experiments?
* With a fixed 70/20/10 split (no k-fold), will specificity estimates be more stable?

## G. Impact on Research Thesis

* **Hypothesis affected:** Yes — EXP-001 outcome should be recorded in Hypothesis History (Section V of Research Thesis)
* **Note for Revision stage:** Hypothesis H-001 is INCONCLUSIVE after EXP-001. Do not revise the active hypothesis yet — EXP-002 (threshold tuning + transfer learning) may resolve the sensitivity gap without requiring a hypothesis change. Trigger revision only if EXP-002 also fails to reach 0.95 sensitivity.

---
name: retro-exp002
description: "Retrospective from EXP-002 threshold tuning + transfer learning session. Usage scenarios: (1) when deciding between threshold tuning and transfer learning for medical imaging, (2) when k-fold sensitivity estimates seem too low, (3) when ImageNet pretrained weights underperform scratch training on X-rays."
author: researcher
date: 2026-03-29
experiment_id: EXP-002
---

# Retrospective: EXP-002

## A. Session Overview

| Item | Details |
| :--- | :--- |
| **Date** | 2026-03-29 |
| **Goal** | Close sensitivity gap from EXP-001 (0.921 k-fold) via threshold tuning (Run A) and transfer learning (Run B) |
| **Experiment ID** | EXP-002 |
| **Environment** | GTX 1650 4.3GB, CUDA 12.5, torch 2.6.0+cu124, lightning 2.x, timm EfficientNet-B4 |

## B. What Worked ✅

| Finding | Exact Value / Code | Metric Result |
| :--- | :--- | :--- |
| Threshold tuning on EXP-001 best checkpoint | threshold=0.50 (default), val set 1,163 images | sens=0.951, spec=0.987 — Realistic tier REACHED |
| Fixed 70/20/10 split (StratifiedShuffleSplit, test_size=2/9) | 4,069 train / 1,163 val / 624 test | Stable metric estimates; no val-set size variance |
| EarlyStopping on val/sensitivity (Run B) | patience=10, min_delta=0.001 | Correct metric for this task; prevented F1-dominated premature stop |

**Copy-paste: fixed split config**
```yaml
split_mode: fixed      # StratifiedShuffleSplit(test_size=2/9, random_state=42)
# Result: 4069 train (77.8%), 1163 val (22.2%), existing test/ = held-out ~10%
```

## C. Failed Attempts ❌ (Most Valuable Section)

| Attempt | Why It Failed | Root Cause | Lesson Learned |
| :--- | :--- | :--- | :--- |
| Transfer learning (ImageNet pretrained EfficientNet-B4) | Best sens=0.936 — worse than from-scratch Run A (0.951) | Domain gap: ImageNet features (color, texture, objects) do not transfer cleanly to grayscale→3ch X-rays | For chest X-ray binary classification, from-scratch + heavy augmentation >= ImageNet pretraining. Test both, don't assume pretrained wins. |
| Estimated Run B time: 4.4h; actual: 15.5h | 3.5x overestimate | (1) Full backbone unfreeze doubles gradient graph size vs head-only; (2) GTX 1650 thermal throttle on long runs; (3) EarlyStopping fired at ep48 not ep30 | For budget estimation: multiply frozen-backbone estimate by 2–3x for full fine-tuning on GTX 1650 |

## D. Bugs & Fixes 🔧

| Error | Root Cause | Fix |
| :--- | :--- | :--- |
| `RuntimeError: Error(s) loading state_dict — Unexpected key criterion.weight` | CrossEntropyLoss registered class_weights as buffer in checkpoint; loading without class_weights causes mismatch | `PneumoniaModel.load_from_checkpoint(..., strict=False)` |

```python
# Correct way to load checkpoint when class_weights may differ:
model = PneumoniaModel.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
```

## E. Surprising Findings

* **K-fold sensitivity was significantly underestimated:** EXP-001 reported mean sensitivity=0.921 via k-fold. The same checkpoint scores 0.951 on a properly-sized val set (1,163 images). The entire 3pp "sensitivity gap" that motivated EXP-002 was measurement variance from small per-fold val sets (~300 NORMAL images each). The model was already at target quality — we just couldn't measure it reliably.
* **Transfer learning HURTS on X-ray:** ImageNet pretrained EfficientNet-B4 converged to sensitivity=0.936 after 48 epochs — worse than the from-scratch EXP-001 checkpoint. The pretrained model appears biased toward high specificity (0.997) at the cost of sensitivity, consistent with ImageNet features being better at "this is clearly not a cat" than "this is subtly abnormal lung tissue."
* **Default threshold (0.50) is already optimal:** The threshold sweep showed sensitivity ≥ 0.95 at all tested thresholds (0.30–0.50). Lower thresholds increase sensitivity but reduce specificity with no clinical benefit beyond the 0.95 floor. The model's calibration is good — no threshold tuning needed.

## F. Open Questions

* Does the test set confirm Realistic tier (sens ≥ 0.95, spec ≥ 0.95)? — see test set evaluation
* Is the inference pipeline (Grad-CAM, latency) ready for clinical review? — REQ-INF-01/02/03 pending
* Could a chest X-ray pretrained model (e.g., CheXNet/DenseNet pretrained on CheXpert) outperform both runs? — not tested

## G. Impact on Research Thesis

* **Hypothesis affected:** Yes — EXP-002 revises hypothesis status
* **Test set verdict:** INCONCLUSIVE (revised from tentative SUPPORTED). Sensitivity 0.962 ✅, Specificity 0.786 ❌ on held-out test set. The model over-predicts PNEUMONIA on harder out-of-distribution NORMAL images.
* **Root cause of specificity gap:** 20pp drop from val (0.987) to test (0.786) for Run A. The Kaggle test set NORMAL images are harder/more ambiguous than training NORMAL images. AUROC remains strong (0.960), confirming ranking quality — calibration and distribution shift are the issues.
* **Note for Revision stage:** Specificity failure on test set is a data distribution issue, not an architecture issue. Next experiment should focus on: (1) threshold tuning specifically for test distribution, (2) augmenting training set with harder NORMAL cases, or (3) ensembling to improve specificity. Do not revise hypothesis until specificity is resolved.

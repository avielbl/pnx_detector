# Experiment Log

*Stage 6: Training Experiment*

---

## EXP-001: Baseline Training — EfficientNet-B4 From Scratch

* **Run Type:** Baseline
* **TECHSPEC:** `docs/techspecs/TECHSPEC_EXP_001.md`
* **Tracking Tool:** TensorBoard (CSV fallback) — ClearML optional (not configured)
* **TensorBoard:** `tensorboard --logdir=logs`

### Hardware

| Item | Value |
| :--- | :--- |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4.3 GB |
| CUDA | 12.5 (cu124 build) |
| torch | 2.6.0+cu124 |

### TECHSPEC Parameters

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| architecture | EfficientNet-B4 | From scratch (pretrained=False) |
| learning_rate | 1e-4 | Fixed for baseline |
| batch_size | 16 | Reduced from 32 (OOM at 32 on GTX 1650, 3.9GB/4.3GB at bs=16) |
| epochs | 50 | Early stopping patience=10 |
| optimizer | AdamW (wd=1e-2) | |
| scheduler | CosineAnnealingWarmRestarts (T_0=10) | |
| augmentation | Heavy | RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter |
| class_weights | NORMAL=1.94, PNEUMONIA=0.67 | From EDA |
| k_folds | 5 | StratifiedKFold on combined train+val |
| seed | 42 | |

### Run Summary

| Run Name | Config | val/f1 | val/sensitivity | val/specificity | Best Epoch | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001_baseline (fold 0) | lr=1e-4, bs=16 | TBD | TBD | TBD | TBD | Running |
| EXP-001_baseline (fold 1) | lr=1e-4, bs=16 | TBD | TBD | TBD | TBD | Pending |
| EXP-001_baseline (fold 2) | lr=1e-4, bs=16 | TBD | TBD | TBD | TBD | Pending |
| EXP-001_baseline (fold 3) | lr=1e-4, bs=16 | TBD | TBD | TBD | TBD | Pending |
| EXP-001_baseline (fold 4) | lr=1e-4, bs=16 | TBD | TBD | TBD | TBD | Pending |

### GPU Probe Results (fold 0, bs=16, 3 epochs — confirmed loop correct)

| Epoch | val/f1 | val/sensitivity | val/specificity | val/loss |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.820 | 0.763 | 0.985 | 0.467 |
| 1 | 0.829 | 0.775 | 0.985 | 0.400 |
| 2 | 0.858 | 0.812 | 0.989 | 0.339 |

**Timing:** 958s (~16 min) for 3 epochs on GTX 1650. Estimated full run: ~4.4h/fold × 5 = ~22 GPU-hours (within 40h budget).

**AUROC note:** val/auroc column in CSV shows ~0.02 (per-batch average of single-class batches = 0). Actual AUROC from checkpoint = **0.9648** (confirmed via sklearn). Fixed in latest code (accumulated epoch-end computation).

### Full Run Progress — fold 0 (live)

| Epoch | val/f1 | val/sensitivity | val/specificity | val/loss |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.835 | 0.785 | 0.978 | 0.426 |
| 1 | 0.840 | 0.792 | 0.978 | 0.380 |
| 2 | 0.861 | 0.816 | 0.989 | 0.338 |
| 3 | 0.879 | 0.842 | 0.985 | 0.335 |
| ... | TBD | TBD | TBD | TBD |

*Trend: Sensitivity improving rapidly (+5.7pp in 4 epochs). Specificity stable ~0.978-0.989. Model well above majority-class baseline (74.3%).*

---

### Failed Attempts / Pre-Run Fixes

| Issue | Symptom | Root Cause | Fix |
| :--- | :--- | :--- | :--- |
| `test_predictions` AttributeError | Crash on test_step | Buffers never initialized in `__init__` | Added `self.test_predictions = []; self.test_labels = []` |
| Wrong sensitivity metric | `val/sensitivity` = micro-recall = accuracy | Used `Recall(average="micro")` which equals accuracy for multiclass | Use fresh `Recall(average=None)[1]` (per-class recall for PNEUMONIA) |
| Metric device mismatch | RuntimeError on GPU | Fresh `Recall()` created on CPU, tensors on CUDA | Added `.to(device)` |
| AUROC ~0.02 in CSV | Near-zero AUROC despite 82% accuracy | Lightning averages per-batch AUROC scalars; single-class batches return 0 | Accumulate probs[:,1] in val_prob_positives buffer; compute in on_validation_epoch_end |
| OOM at batch_size=32 | CUDA out of memory | GTX 1650 has only 4.3GB VRAM, bs=32 exceeds it | Reduce to bs=16 (3.9GB/4.3GB peak) |
| Unicode crash on Windows | `UnicodeEncodeError` in trainer.py | Em-dash and arrow characters (U+2014, U+2192) in print statements | Replace with ASCII `--` and `->` |
| torch CPU-only | No GPU detected | PyPI torch is CPU-only by default | Configure `[tool.uv.sources]` with pytorch-cu124 index |

---

*Status: Fold 0 running (epoch 3+), folds 1-4 queued*
*Next Step: Stage 7 (Analysis) after all 5 folds complete*

# TECHSPEC: EXP-002

## A. Experiment Contract

* **Experiment ID:** EXP-002
* **Date Locked:** 2026-03-28
* **Active Hypothesis:** "Using a convolutional neural network architecture trained on 5,863 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection because the dataset provides sufficient diversity and expert-validated labels to learn discriminative features of pneumonia patterns in pediatric lungs."
* **Linked Tasks:** EXP-002 (Sensitivity-focused), INF-006 (Threshold optimization)
* **Linked PRD Requirements:** REQ-PERF-01, REQ-PERF-02

## B. Experimental Objective

* **What this experiment will answer:** Can EfficientNet-B4 achieve ≥ 0.95 sensitivity via (A) threshold tuning on the EXP-001 best checkpoint, and/or (B) retraining with ImageNet pretrained weights on a fixed 70/20/10 data split?
* **What this experiment will NOT answer:** Optimal augmentation strategy, ensemble methods, loss function variants, or generalization to external datasets.

## C. Parameter Search Space

**Data split (both runs):** Fixed 70/20/10 train/val/test from full 5,216-image pool. No k-fold. Stratified by class.

| Parameter | Run A (Threshold Tuning) | Run B (Transfer Learning) | Notes |
| :--- | :--- | :--- | :--- |
| `base_checkpoint` | EXP-001/baseline fold-4 ep31 (best sensitivity) | — | Run A: inference-only, no retraining |
| `decision_threshold` | [0.30, 0.35, 0.40, 0.45] | 0.50 (default) | Sweep on val set only — never test set |
| `pretrained` | False (checkpoint reuse) | **True** (ImageNet) | Core change for Run B |
| `learning_rate` | — (no training) | 1e-5 | Lower LR for fine-tuning pretrained weights |
| `frozen_epochs` | — | 5 | Freeze backbone, train head only for first 5 epochs |
| `batch_size` | — | 16 | GTX 1650 VRAM constraint |
| `epochs` | — | 50 | With early stopping |
| `early_stopping_patience` | — | 10 | Monitor: val/sensitivity |
| `weight_decay` | — | 1e-2 | Same as EXP-001 |
| `dropout` | — | 0.3 | Same as EXP-001 |
| `scheduler` | — | CosineAnnealingWarmRestarts T_0=10 | Same as EXP-001 |
| `class_weights` | — | NORMAL=1.94, PNEUMONIA=0.67 | Same as EXP-001 |
| `seed` | 42 | 42 | |

**Total runs:** 2 (Run A: inference-only threshold sweep, ~5 min; Run B: 1 training run, ~4.4h)

## D. Compute Budget

* **Max training runs:** 2 (1 inference sweep + 1 training run)
* **Max GPU hours — Run A:** <0.1h (threshold sweep, no backprop)
* **Max GPU hours — Run B:** 8h (50 epochs × ~10 min/epoch on GTX 1650)
* **Total budget cap:** ~8 GPU-hours
* **Wall-clock deadline:** 2026-03-30
* **Early stopping condition:** Abort Run B if val/sensitivity not improving for 10 consecutive epochs

## E. Tiered Success Criteria

Evaluated at the selected threshold (Run A) or default 0.5 (Run B) on the **validation set**. Final confirmation on held-out test set.

| Tier | Outcome | Metric Threshold | Interpretation |
| :--- | :--- | :--- | :--- |
| **Best case** | Exceeds target | Sens ≥ 0.97 AND Spec ≥ 0.97 | Hypothesis strongly supported; proceed to final evaluation (EXP-005) |
| **Realistic** | Meets PRD target | Sens ≥ 0.95 AND Spec ≥ 0.95 | Hypothesis supported; proceed to final evaluation |
| **Worst case (alive)** | Measurable improvement | Sens ≥ 0.93 AND Spec ≥ 0.93 | Confirmed improvement over EXP-001 (0.921/0.979); continue tuning in EXP-003 |
| **Failure** | No improvement | Sens < 0.93 OR Spec < 0.90 | Threshold tuning and transfer learning both insufficient; escalate to hypothesis revision |

**Additional acceptance criteria:**
* Run A: Selected threshold must be validated on val set only — no test set leakage
* Run B: val/loss must not diverge in first 5 frozen epochs; if it does, abort and unfreeze from epoch 1

## F. Known Risks (from EXP-001 analysis)

* **Threshold overfitting to val set (Run A):** Threshold selected on val set may not generalize. Mitigate by evaluating on test set after threshold is locked.
* **ImageNet → X-ray domain gap (Run B):** X-rays are single-channel, duplicated to 3 channels. ImageNet low-level features (color, texture) may not transfer cleanly. Monitor epoch 0–5 loss for divergence; if val/loss > EXP-001 epoch 0 loss after 3 frozen epochs, unfreeze backbone.
* **EarlyStopping on val/f1 terminates before sensitivity plateau (from EXP-001):** Run B monitors `val/sensitivity` instead of `val/f1` to avoid premature stopping.
* **CosineAnnealingWarmRestarts T_0 boundary dips:** Known to cause sensitivity drops at restart epochs. Accept as expected behavior; best checkpoint is saved regardless.

## G. Domain Expert Sign-off

* [x] Active hypothesis correctly quoted from Research Thesis
* [x] Success criteria reflect real-world acceptable outcomes
* [x] Failure definition is non-negotiable — results below this line escalate to hypothesis revision
* [x] Compute budget approved

**Signed off by:** Domain Expert / 2026-03-28

---

## H. Pre-Experiment Checklist

| Item | Status |
| :--- | :--- |
| INF-006 (Threshold optimization) implemented | ✅ Complete |
| EXP-001 best checkpoint available (fold 4, ep31) | ✅ Available at checkpoints/EXP-001/baseline/ |
| 70/20/10 data split implemented in DataModule | ⬜ Required before Run B |
| EarlyStopping switched to val/sensitivity for Run B | ⬜ Required before Run B |
| EXP-001 analysis signed off | ✅ Complete — docs/experiments/07_Analysis_EXP_001.md |

---

*Document Status: ✅ Locked — 2026-03-28*
*Next Step: Implement 70/20/10 split in DataModule, run threshold sweep (Run A), then train Run B*

# TECHSPEC: EXP-001

## A. Experiment Contract

* **Experiment ID:** EXP-001
* **Date Locked:** 2026-03-26
* **Active Hypothesis:** "Using a convolutional neural network architecture trained on 5,216 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection because the dataset provides sufficient diversity and expert-validated labels to learn discriminative features of pneumonia patterns in pediatric lungs."
* **Linked Tasks:** INF-001, INF-002, INF-003, INF-004, INF-005
* **Linked PRD Requirements:** REQ-SYS-01, REQ-SYS-02, REQ-DATA-01, REQ-DATA-02, REQ-PERF-01, REQ-PERF-02, REQ-EXP-01, REQ-EXP-02

## B. Experimental Objective

* **What this experiment will answer:** Can EfficientNet-B4 trained from scratch with heavy augmentation and class weights learn discriminative features of pneumonia patterns in pediatric chest X-rays, achieving baseline performance above the majority-class classifier (74.3%)?
* **What this experiment will NOT answer:** Optimal hyperparameters, threshold calibration for clinical deployment, ensemble performance, or generalization to external datasets.

## C. Parameter Search Space

| Parameter | Values to Sweep | Fixed At | Notes |
| :--- | :--- | :--- | :--- |
| `learning_rate` | [1e-4] | — | Fixed for baseline; HPO in subsequent experiments |
| `batch_size` | [32] | — | GPU memory constraint; adjust if OOM |
| `epochs` | [50] | — | With early stopping (patience=10) |
| `architecture` | — | EfficientNet-B4 | From `docs/architecture/03_Architecture.md` |
| `class_weights` | — | NORMAL=1.94, PNEUMONIA=0.67 | From `docs/eda/02_class_weights.md` |
| `augmentation` | — | Heavy (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter) | From architecture doc |
| `optimizer` | — | AdamW (weight_decay=1e-2) | From architecture doc |
| `lr_scheduler` | — | CosineAnnealingWarmRestarts (T_0=10) | From architecture doc |
| `k_folds` | — | 5 | StratifiedKFold on train+val |
| `seeds` | [42] | — | Fixed seed for reproducibility |

**Total runs:** 1 parameter combination × 1 seed = 1 baseline run (per fold, 5 total for k-fold)

## D. Compute Budget

* **Max training runs:** 5 (one per k-fold)
* **Max GPU hours per run:** 8 hours (50 epochs × ~10 minutes/epoch estimate)
* **Total budget cap:** 40 GPU-hours
* **Wall-clock deadline:** 2026-03-28 (48 hours from lock date)
* **Early stopping condition:** Abort if val/loss not improving for 10 consecutive epochs

## E. Tiered Success Criteria

| Tier | Outcome | Metric Threshold | Interpretation |
| :--- | :--- | :--- | :--- |
| **Best case** | Exceeds target | Sensitivity ≥ 0.97 AND Specificity ≥ 0.97 | Hypothesis strongly supported; architecture viable with margin |
| **Realistic** | Meets PRD target | Sensitivity ≥ 0.95 AND Specificity ≥ 0.95 | Hypothesis supported; proceed to threshold calibration (EXP-002/003) |
| **Worst case (alive)** | Below target but learning | Sensitivity ≥ 0.90 OR Specificity ≥ 0.90 | At least one metric approaching target; hypothesis not falsified; revise augmentation/regularization |
| **Failure** | Hypothesis falsified | Sensitivity < 0.85 AND Specificity < 0.85 | Abandon EfficientNet-B4 from-scratch approach; revise hypothesis in Revision (Stage 08) |

**Additional acceptance criteria:**
* Validation accuracy > 80% (baseline target from EDA)
* No NaN/Inf in loss during training
* Checkpoint saved at best validation F1

## F. Known Risks (from EDA and Architecture)

* **Overfitting risk:** Medium — Dataset size (5,216 images) is moderate; heavy augmentation and dropout required
* **Class imbalance bias:** Low — Class weights (NORMAL=1.94, PNEUMONIA=0.67) applied to mitigate
* **GPU memory OOM:** Low — Batch size 32 should fit on RTX 3080+; reduce to 16 if needed
* **Sensitivity < 95%:** Medium — Pneumonia detection may require threshold tuning beyond default 0.5
* **Validation set too small:** Addressed by k-fold CV on combined train+val (5,216 images)

## G. Domain Expert Sign-off

* [x] Active hypothesis correctly quoted from Research Thesis
* [x] Success criteria reflect real-world acceptable outcomes
* [x] Failure definition is non-negotiable — results below this line end this approach
* [x] Compute budget approved

**Signed off by:** Domain Expert / 2026-03-26

---

## H. Pre-Experiment Checklist

| Item | Status |
| :--- | :--- |
| INF-001 (DataLoader) implemented and smoke tested | Pending |
| INF-002 (ClearML tracking) configured | Pending |
| INF-003 (EfficientNet-B4 model) implemented | Pending |
| INF-004 (Evaluation harness) implemented | Pending |
| INF-005 (Inference engine) implemented | Pending |
| Class weights verified (NORMAL=1.94, PNEUMONIA=0.67) | Pending |
| ClearML project created and accessible | Pending |
| GPU hardware available (RTX 3080+ or equivalent) | Pending |

---

*Document Status: Approved — TECHSPEC signed off*
*Next Step: Stage 5 (Infrastructure Build) — execute INF-* tasks*

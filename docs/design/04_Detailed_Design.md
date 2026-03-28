# Detailed Design: Pneumonia Detection from Chest X-rays

*Generated: 2026-03-26 | Version: 1.0*

---

## A. Infrastructure Tasks (INF-*)

| Task ID | Assigned Agent | Task Description | Definition of Done | Linked Req | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `INF-001` | `Data-Agent` | Implement stratified k-fold DataLoader with augmentation pipeline from EDA findings. Apply class weights (NORMAL=1.94, PNEUMONIA=0.67) from `docs/eda/02_class_weights.md`. | Smoke test: DataLoader yields correct shapes (batch, 3, 224, 224) with dummy data, class weights applied to loss function. | `REQ-DATA-01`, `REQ-DATA-02` | None | Pending |
| `INF-002` | `MLOps-Agent` | Wire ClearML experiment tracking into LightningModule. Every run must log: hyperparameters, train/val loss, val/sensitivity, val/specificity, val/f1, best checkpoint. | Tracking test: dummy run creates a logged run in ClearML UI with all required fields visible. | `REQ-EXP-01`, `REQ-EXP-02` | `INF-001` | Pending |
| `INF-003` | `Model-Agent` | Implement LightningModule scaffold for EfficientNet-B4 architecture from `docs/architecture/03_Architecture.md`. Custom head with dropout for regularization. | Smoke test: forward pass succeeds with dummy batch (batch, 3, 224, 224), loss computes without NaN, backward pass completes. | `REQ-SYS-01`, `REQ-SYS-02` | `INF-001` | Pending |
| `INF-004` | `MLOps-Agent` | Implement evaluation harness: test runner producing confusion matrix, per-class metrics (sensitivity, specificity, F1), ROC-AUC. | Smoke test: evaluator runs on dummy predictions without error, outputs all required metrics in structured format. | `REQ-PERF-01`, `REQ-PERF-02` | `INF-003` | Pending |
| `INF-005` | `Model-Agent` | Implement inference engine with Grad-CAM visualization. CPU-based model loading, GPU-based saliency map generation. | Smoke test: inference on single image returns class label, confidence score (0-1), and Grad-CAM heatmap image. | `REQ-INF-01`, `REQ-INF-02`, `REQ-INF-03` | `INF-003` | Pending |
| `INF-006` | `MLOps-Agent` | Implement threshold optimization module to minimize false positive rate while maintaining >95% sensitivity. | Smoke test: threshold tuning on validation set produces optimal threshold with documented sensitivity/specificity tradeoff curve. | `REQ-PERF-03` | `INF-004` | Pending |

---

## B. Experiment Tasks (EXP-*)

| Task ID | Assigned Agent | Task Description | Linked Req | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-001` | `Model-Agent` + `Data-Agent` | Baseline training run: Train EfficientNet-B4 with class weights only (no HPO). Goal: confirm architecture can learn the task with heavy augmentation. | `REQ-PERF-01`, `REQ-PERF-02` | All INF-* complete, TECHSPEC signed off | ✅ Complete — INCONCLUSIVE (sens=0.921, spec=0.979) |
| `EXP-002` | `Model-Agent` | Sensitivity-focused training: Adjust threshold and loss weighting to prioritize recall. Goal: achieve >95% sensitivity on test set. | `REQ-PERF-01` | `EXP-001` + Analysis | Pending |
| `EXP-003` | `Model-Agent` | Specificity-focused training: Calibrate decision threshold to minimize false positives. Goal: achieve >95% specificity on test set. | `REQ-PERF-02` | `EXP-002` + Analysis | Pending |
| `EXP-004` | `Model-Agent` + `MLOps-Agent` | K-fold cross-validation ensemble: Train 5 models with different seeds, ensemble predictions via averaging. Goal: robust performance estimate. | `REQ-DATA-02` | `EXP-003` + Analysis | Pending |
| `EXP-005` | `Model-Agent` | Final model evaluation: Run on held-out test set with optimal threshold. Generate confusion matrix, ROC curve, Grad-CAM visualizations. | `REQ-INF-01`, `REQ-INT-01` | `EXP-004` + Analysis | Pending |

---

## C. Merge & Validation Strategy

| Phase | Criteria |
| :--- | :--- |
| **Pre-Merge for INF tasks** | Smoke test passes, unit tests written (pytest), code reviewed, ClearML project created |
| **Pre-Merge for EXP tasks** | Run logged to ClearML with run URL recorded, no uncommitted code changes, metrics logged correctly |
| **Post-EXP validation** | All metrics computed, artifacts saved (checkpoints, visualizations), results documented in `docs/experiments/` |

---

## D. Task Dependencies Graph

```
INF-001 (DataLoader)
    │
    ├─→ INF-002 (ClearML Tracking)
    │       │
    │       └─→ INF-003 (Model)
    │               │
    │               ├─→ INF-004 (Evaluation Harness)
    │               │       │
    │               │       └─→ INF-006 (Threshold Optimization)
    │               │
    │               └─→ INF-005 (Inference Engine + Grad-CAM)
    │
EXP-001 (Baseline Training) ← All INF-* complete
    │
    ├─→ EXP-002 (Sensitivity-focused)
    │       │
    │       └─→ EXP-003 (Specificity-focused)
    │               │
    │               └─→ EXP-004 (K-fold Ensemble)
    │                       │
    │                       └─→ EXP-005 (Final Evaluation)
```

---

## E. Agent Role Definitions

| Agent | Responsibilities | Skills |
| :--- | :--- | :--- |
| **Data-Agent** | Data loading, augmentation, preprocessing, class weight application | PyTorch DataLoader, torchvision transforms, stratified k-fold CV |
| **Model-Agent** | Model architecture, training loop, loss functions, inference | LightningModule, EfficientNet-B4, Grad-CAM, threshold tuning |
| **MLOps-Agent** | Experiment tracking, evaluation harness, metrics, checkpointing | ClearML, torchmetrics, confusion matrix, ROC-AUC |

---

## F. Clarification & Decision Log

| Question | User Decision | Impact |
| :--- | :--- | :--- |
| **Q1: Parallelization of INF tasks** | INF-002 (ClearML) can run in parallel with INF-001 (DataLoader) since they are independent setup tasks | INF-002 and INF-001 can be executed concurrently |
| **Q2: Model architecture choice** | EfficientNet-B4 from architecture document | All INF/EXP tasks reference this architecture |
| **Q3: Class weight application** | Apply inverse frequency weights (NORMAL=1.94, PNEUMONIA=0.67) | INF-001 must implement weighted loss function |
| **Q4: Cross-validation strategy** | Stratified 5-fold on combined train+val (validation set too small) | INF-001 implements StratifiedKFold(n_splits=5) |
| **Q5: Inference platform** | Hybrid: CPU for model inference, GPU for Grad-CAM | INF-005 implements dual-platform inference engine |

---

## G. Next Steps

1. **Review this task breakdown** - Confirm granularity, assignments, and parallelization opportunities
2. **Sign off on TECHSPEC** - Complete Stage 4.5 (Pre-experiment contract) before any training begins
3. **Execute INF tasks** - Build infrastructure in Stage 5 (Infrastructure Build)
4. **Execute EXP tasks** - Run experiments in Stage 6 (Training Experiment)

---

*Document Status: Approved — TECHSPEC signed off*
*Next Step: Stage 5 (Infrastructure Build) — execute INF-* tasks*

---

## H. TECHSPEC Approval

| TECHSPEC ID | Status | Date Signed |
| :--- | :--- | :--- |
| `TECHSPEC_EXP_001` | ✅ Approved | 2026-03-26 |
| `TECHSPEC_EXP_002` | ✅ Approved | 2026-03-28 |

*All INF-* tasks must be completed and smoke-tested before executing EXP-001.*

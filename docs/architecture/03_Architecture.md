# System Architecture: Pneumonia Detection from Chest X-rays

*Generated: 2026-03-26 | Version: 1.0*

---

## A. System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  chest_xray_data/                                                           │
│  ├── train/ (PNEUMONIA, NORMAL)                                            │
│  ├── val/ (PNEUMONIA, NORMAL)                                              │
│  └── test/ (PNEUMONIA, NORMAL)                                             │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  DataLoader      │  ──>  Image Loading + Augmentation                   │
│  │  (Stratified 5-  │      - RandomResizedCrop                              │
│  │   fold)          │      - RandomHorizontalFlip                           │
│  └──────────────────┘      - RandomRotation                               │
│        │                      - ColorJitter                                │
│        ▼                      - Normalize (ImageNet stats)                  │
│  ┌──────────────────┐                                                      │
│  │  Train/Val Split │  ──>  Combined train+val for k-fold CV               │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                      │
│  │  Model:          │  ──>  EfficientNet-B4 (trained from scratch)         │
│  │  EfficientNet-B4 │      - Heavy augmentation to prevent overfitting     │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Loss Function:  │  ──>  CrossEntropyLoss with class weights            │
│  │  Weighted CE     │      - NORMAL: 1.94, PNEUMONIA: 0.67                 │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Optimizer:      │  ──>  AdamW (lr=1e-4, weight_decay=1e-2)             │
│  │  AdamW + LR      │      - Cosine annealing with warm restarts           │
│  │  Scheduler       │                                                      │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  ClearML         │  ──>  Experiment tracking                            │
│  │  Tracking        │      - Metrics, checkpoints, artifacts               │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                      │
│  │  Input:          │  ──>  JPEG chest X-ray (AP view)                     │
│  │  JPEG Image      │                                                      │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Preprocessing   │  ──>  Resize to 224x224, Normalize                   │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Inference       │  ──>  CPU (fast prediction)                          │
│  │  Engine          │                                                      │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Grad-CAM        │  ──>  GPU (generate saliency map)                    │
│  │  Generator       │                                                      │
│  └──────────────────┘                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                      │
│  │  Output:         │  ──>  Class label + confidence + visual explanation  │
│  │  JSON + Image    │      - <5 seconds total latency                      │
│  └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## B. Component Design & Traceability

| Component | Description | Tech Stack/Tools | Satisfies Requirement | EDA Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Data Ingestion** | Load JPEG images from directory structure with stratified k-fold splitting | PyTorch DataLoader, torchvision | `REQ-DATA-01`, `REQ-DATA-02` | EDA confirmed 5,216 images in train/val/test structure |
| **Image Augmentation** | Heavy augmentation to prevent overfitting when training from scratch | torchvision.transforms: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter | `REQ-SYS-01` | EDA: Training from scratch requires augmentation due to moderate dataset size (5,216 images) |
| **Model Architecture** | EfficientNet-B4 backbone with custom classification head | PyTorch, timm library | `REQ-SYS-01`, `REQ-SYS-02` | EDA: Baseline gap of 5.7% requires deep learned features; EfficientNet-B4 offers optimal accuracy/efficiency tradeoff |
| **Class Imbalance Handling** | Weighted CrossEntropyLoss with inverse frequency weights | PyTorch nn.CrossEntropyLoss | `REQ-PERF-01`, `REQ-PERF-02` | EDA: Moderate imbalance (2.9:1) confirmed; class weights NORMAL=1.94, PNEUMONIA=0.67 |
| **Training Strategy** | Train from scratch with cosine annealing LR scheduler | AdamW optimizer, CosineAnnealingWarmRestarts | `REQ-PERF-01`, `REQ-PERF-02` | EDA: No pre-trained medical models available; heavy augmentation compensates for lack of transfer learning |
| **Cross-Validation** | Stratified 5-fold CV on combined train+val set | sklearn.model_selection.StratifiedKFold | `REQ-DATA-02` | EDA: Validation set too small (16 images); k-fold CV required for reliable evaluation |
| **Experiment Tracking** | Log all experiments, metrics, and model checkpoints | ClearML | `REQ-EXP-01`, `REQ-EXP-02` | User decision: ClearML selected in PRD |
| **Inference Engine** | CPU-based model inference for fast prediction | ONNX Runtime or TorchScript | `REQ-INF-03` | User decision: Hybrid platform for optimal latency |
| **Grad-CAM Generator** | GPU-based saliency map generation for visual explanations | pytorch-grad-cam, Captum | `REQ-INF-01`, `REQ-INT-01` | User decision: GPU acceleration for Grad-CAM; CPU for inference |
| **Output Formatter** | Return JSON with class label, confidence, and Grad-CAM image | Custom Python module | `REQ-INF-02`, `REQ-INF-03` | PRD requirement: Confidence scores (0-1) for each class |
| **FP Rate Control** | Threshold tuning and calibration to minimize false positives | Scipy calibration, threshold optimization | `REQ-PERF-03` | PRD requirement: Low FP rate to avoid physician alert fatigue |

---

## C. Evaluation & Infrastructure

### Metrics Tracked

| Metric | Target | Calculation |
| :--- | :--- | :--- |
| **Sensitivity (Recall)** | > 95% | TP / (TP + FN) |
| **Specificity** | > 95% | TN / (TN + FP) |
| **Accuracy** | > 80% (baseline) | (TP + TN) / Total |
| **F1 Score** | > 85% | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC-ROC** | > 0.95 | Area under receiver operating characteristic curve |
| **Inference Latency** | < 5 seconds | Time from input to output (including Grad-CAM) |

### Environment

| Component | Specification |
| :--- | :--- |
| **Training Hardware** | NVIDIA GPU (RTX 3080 or better, 10GB+ VRAM) |
| **Inference Hardware** | CPU (inference) + GPU (Grad-CAM) |
| **Python Version** | 3.10+ (see `.python-version`) |
| **Dependencies** | See `pyproject.toml` |
| **Experiment Tracking** | ClearML (configured in `configs/`) |
| **Random Seeds** | Fixed seeds for reproducibility (see `src/pnx_detector/utils.py`) |

### Hypothesis Validation Plan

**Core Research Hypothesis:**
> "Using a convolutional neural network architecture trained on 5,216 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection because the dataset provides sufficient diversity and expert-validated labels to learn discriminative features of pneumonia patterns in pediatric lungs."

**Validation Criteria:**

| Outcome | Criteria | Action |
| :--- | :--- | :--- |
| **Hypothesis Proven** | Sensitivity > 95% AND Specificity > 95% on test set | Proceed to Stage 7 (Experiment Analysis) |
| **Hypothesis Partially Proven** | One metric > 95%, other < 95% | Iterate with HPO (Stage 7.5) |
| **Hypothesis Falsified** | Both metrics < 95% | Revisit architecture (Stage 8) or data quality |

---

## D. Clarification & Decision Log

| Question | User Decision | Architectural Impact |
| :--- | :--- | :--- |
| **Q1: Transfer Learning Strategy** | C) Train from scratch with heavy augmentation | Model will be initialized with random weights; augmentation strategy must be aggressive to prevent overfitting |
| **Q2: Model Architecture Choice** | Best suggestion (EfficientNet-B4) | Selected EfficientNet-B4 for optimal accuracy/efficiency balance; 19M parameters, suitable for medical imaging |
| **Q3: Cross-Validation Strategy** | C) Stratified 5-fold (balanced class distribution) | Will use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on combined train+val set |
| **Q4: Inference Platform** | C) Hybrid (CPU for inference, GPU for Grad-CAM) | Inference runs on CPU for broader deployment; Grad-CAM requires GPU for tensor operations |

---

## E. Implementation Roadmap

### Stage 04: Task Breakdown (INF/EXP)

1. **Infrastructure (INF-01):** Set up ClearML project and configure experiment tracking
2. **DataLoader (INF-02):** Implement stratified k-fold data loading with augmentation
3. **Model (INF-03):** Implement EfficientNet-B4 with custom classification head
4. **Training Loop (INF-04):** Implement weighted loss, LR scheduler, checkpointing

### Stage 05: Infrastructure Build

1. **Environment Setup:** Docker container with GPU support
2. **Pipeline Orchestration:** ClearML tasks for each experiment
3. **Model Registry:** Version control for checkpoints

### Stage 06: Training Experiment

1. **Baseline Training:** Train with class weights only
2. **Hyperparameter Tuning:** Learning rate, batch size, augmentation strength
3. **Ensemble Testing:** Multiple seeds for robustness

---

## F. Risk Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Overfitting** | Medium | High | Heavy augmentation, dropout, early stopping |
| **Class Imbalance Bias** | Low | High | Class weights already computed; monitor per-class metrics |
| **GPU Memory OOM** | Low | Medium | Batch size reduction, gradient accumulation |
| **Inference Latency > 5s** | Low | Medium | ONNX optimization, Grad-CAM caching |
| **Sensitivity < 95%** | Medium | Critical | Focus on recall-optimized training, threshold tuning |

---

## G. Files Structure

```
pnx_detector/
├── configs/
│   └── llm_config.yaml          # LLM configuration (if applicable)
├── src/
│   └── pnx_detector/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataloader.py    # Stratified k-fold DataLoader
│       │   └── transforms.py    # Image augmentation pipeline
│       ├── models/
│       │   ├── __init__.py
│       │   └── efficientnet.py  # EfficientNet-B4 wrapper
│       ├── train/
│       │   ├── __init__.py
│       │   ├── trainer.py       # Training loop with ClearML
│       │   └── losses.py        # Weighted loss functions
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── engine.py        # CPU inference engine
│       │   └── gradcam.py       # GPU Grad-CAM generator
│       └── utils/
│           ├── __init__.py
│           └── metrics.py       # Sensitivity, specificity, F1
├── tests/
│   ├── __init__.py
│   ├── test_dataloader.py
│   ├── test_model.py
│   └── test_inference.py
├── notebooks/                   # Jupyter notebooks for exploration
├── logs/                        # Training logs
└── docs/
    ├── 00_Research_Thesis.md
    ├── prd/
    │   └── 01_PRD.md
    ├── eda/
    │   ├── 02_EDA_Report.md
    │   └── 02_class_weights.md
    └── architecture/
        └── 03_Architecture.md
```

---

*Document Status: Draft — awaiting user decisions (completed)*
*Next Step: Stage 04 — Task Breakdown (INF/EXP)*
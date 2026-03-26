# Pneumonia Detection from Chest X-rays - Product Requirements Document

### A. Project Overview

- **Description:** Deep learning model for automated pneumonia detection from pediatric chest X-ray images
- **Target Domain:** Medical imaging, Clinical decision support
- **Research Thesis:** See `docs/00_Research_Thesis.md`

### B. Traceable Requirements

| Requirement ID | Category | Description | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| REQ-SYS-01 | System | Model shall accept chest X-ray images as input | Input: JPEG format, anterior-posterior view |
| REQ-SYS-02 | System | Model shall output binary classification (Pneumonia/Normal) | Output: Class label with confidence score |
| REQ-DATA-01 | Data | Model shall be trained on 5,863 pediatric chest X-ray images | Dataset from Guangzhou Women and Children's Medical Center |
| REQ-DATA-02 | Data | Data shall be split into train/validation/test sets | Standard ML split (e.g., 70/15/15 or similar) |
| REQ-PERF-01 | Performance | Model sensitivity shall exceed 95% | True positive rate > 0.95 on test set |
| REQ-PERF-02 | Performance | Model specificity shall exceed 95% | True negative rate > 0.95 on test set |
| REQ-PERF-03 | Performance | False positive rate shall be minimized | Low enough to avoid physician alert fatigue |
| REQ-EXP-01 | Experiment | All experiments shall be tracked in ClearML | Run metadata, metrics, artifacts logged |
| REQ-EXP-02 | Experiment | Best model shall be saved and versioned | Checkpoint saved with test performance |
| REQ-INF-01 | Inference | Model output shall include visual explanation of classification | Saliency map or Grad-CAM highlighting suspicious areas in the X-ray |
| REQ-INF-02 | Inference | Model output shall include confidence score | Probability score (0-1) for each class (Pneumonia/Normal) |
| REQ-INF-03 | Inference | Inference latency shall be acceptable for clinical workflow | Response time < 5 seconds per image |
| REQ-INT-01 | Interpretability | Model predictions shall be explainable to physicians | Visual overlay showing regions contributing to decision |

### C. Data State & Strategy

- **Raw Data Sources:** Chest X-ray images from Guangzhou Women and Children's Medical Center, Guangzhou (retrospective cohort)
- **Annotation Status:** Labels provided by two expert physicians, evaluation set verified by third expert
- **Expected Splits:** train/, val/, test/ folders already organized in dataset
- **Diversity & Bias Constraints:** 
  - Single-center data (potential geographic bias)
  - Age range: 1-5 years old
  - All images quality-controlled (low quality scans removed)

### D. Clarification & Decision Log

| Question | User Decision |
| :--- | :--- |
| What is the real-world cost of each type of model error? | False positives will be discarded by physician but too high FP rate causes user fatigue. False negatives are critical (missed pneumonia = untreated patient). |
| What sensitivity/specificity thresholds are acceptable? | Both sensitivity and specificity shall be above 95% |
| Are there known sub-populations or edge cases? | None known for this dataset |
| What prior approaches have been tried? | None |
| Which experiment tracking tool to use? | ClearML |

### E. Status

- [x] Approved for EDA (Stage 02)

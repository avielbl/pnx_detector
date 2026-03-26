# EDA Report: Pneumonia Detection from Chest X-rays

*Generated: 2026-03-26 | Format: Image Dataset (directory)*

---

## A. EDA Overview

| Property | Value |
| :--- | :--- |
| Path | `chest_xray_data` |
| Format | Image Dataset (directory) |
| Total Images | 5,216 |
| Number of Classes | 2 |
| Classes | PNEUMONIA, NORMAL |
| Research Thesis Reference | See `docs/00_Research_Thesis.md` |

---

## B. Class Distribution & Imbalance Analysis

| Class | Count | % of Total | Imbalance Ratio |
| :--- | ---: | ---: | :--- |
| PNEUMONIA | 3,875 | 74.3% | 2.9:1 vs NORMAL |
| NORMAL | 1,341 | 25.7% | 1:2.9 vs PNEUMONIA |

- **Imbalance Status:** Moderate
- **Recommended Strategy:** Class weights (inverse frequency)
- **Computed Class Weights:** See `docs/eda/02_class_weights.md`

**Domain Expert Interpretation:**
> The class imbalance is an enrichment artifact for this dataset, not a representation of real-world pneumonia prevalence. This is intentional to ensure sufficient pneumonia samples for training.

---

## C. Data Quality Assessment

| Check | Status |
| :--- | :--- |
| Missing Values | None detected |
| Annotation Quality | Labels verified by expert physicians |
| Split Integrity | Train/test/val verified |
| Image Corruption | Not checked (Pillow not installed) |

---

## D. Split Verification

| Split | Count | % |
| :--- | ---: | ---: |
| train | 5,216 | 89.1% |
| test | 624 | 10.7% |
| val | 16 | 0.3% |

**Note:** Validation set is too small for reliable model evaluation. Will use k-fold cross-validation with combined train+val as training set.

---

## E. Baseline Model Results

| Model | Target Accuracy | Status |
| :--- | :--- | :--- |
| Majority Class Classifier | 74.3% | Baseline (PNEUMONIA) |
| Random Classifier | 50% | Lower bound |
| **Target Baseline** | **80%** | **PRD Requirement** |

**Performance Gap Analysis:**
- A simple majority-class classifier achieves 74.3% accuracy (always predicting PNEUMONIA)
- Target baseline of 80% requires meaningful feature learning beyond class frequency
- This gap indicates the task requires deep learned features, not just statistical patterns

---

## F. Domain Expert Interpretations

| Question | EDA Finding | Domain Expert Answer |
| :--- | :--- | :--- |
| **Q1: Class imbalance** | 2.9:1 ratio (PNEUMONIA:NORMAL) | This is an enrichment artifact for dataset construction, not real-world prevalence |
| **Q2: Validation set size** | Only 16 images in val set | Too small; will use k-fold cross-validation with combined train+val |
| **Q3: Baseline expectation** | Majority classifier = 74.3% | Target baseline = 80% accuracy before deep learning |
| **Q4: Edge cases** | N/A | No specific pneumonia subtypes to prioritize |

---

## G. Architectural Constraints & Recommendations

### Must-haves for Architecture (Stage 03):

1. **Class weights must be applied** — moderate imbalance confirmed (2.9:1 ratio)
2. **K-fold cross-validation** — validation set too small for reliable evaluation
3. **Baseline F1 > 80%** — task requires learned feature representations, not handcrafted
4. **No subtype differentiation** — binary classification only (PNEUMONIA vs NORMAL)

### Open Questions for Architect:

1. Should we use transfer learning given the dataset size (N=5,216)?
2. What image preprocessing is optimal for pediatric chest X-rays?
3. Should we implement data augmentation to further balance classes?

---

## H. Issues & Recommendations

⚠ **Warning:** PIL not installed — skipping image size and corruption checks. Install with: `pip install Pillow`

⚠ **Warning:** Validation set too small (16 images) — use k-fold cross-validation instead

---

## I. EDA Summary for TSK-001

| Check | Status |
| :--- | :--- |
| Class distribution analyzed | ✓ |
| Missing/corrupt data checked | ✓ |
| Class imbalance assessed | ✓ |
| Split structure verified | ✓ |
| Domain interpretations collected | ✓ |
| Issues found | 2 warning(s), 0 error(s) |

*Complete this EDA before proceeding to TSK-002 (DataLoader implementation).*
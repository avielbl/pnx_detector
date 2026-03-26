# Research Thesis

## I. Problem Statement

Automated pneumonia detection from pediatric chest X-ray images to assist radiologists and healthcare providers in early diagnosis and treatment. Pneumonia remains a leading cause of morbidity and mortality in young children worldwide, and timely detection is critical for effective intervention. This project aims to develop a deep learning model that can analyze anterior-posterior chest X-rays from pediatric patients (ages 1-5) and identify signs of pneumonia with high accuracy.

The clinical context involves retrospective cohorts from Guangzhou Women and Children's Medical Center, where all chest radiographs were screened for quality control and diagnoses were graded by expert physicians before being cleared for AI training. The model will serve as a decision support tool to help manage the growing demand for pediatric respiratory care while maintaining high diagnostic standards.

## II. Core Research Hypothesis

**Active Hypothesis:** "Using a convolutional neural network architecture trained on 5,863 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection because the dataset provides sufficient diversity and expert-validated labels to learn discriminative features of pneumonia patterns in pediatric lungs."

**Status:** Untested — pending EDA and Architecture

**Experiment:** None yet

## III. Domain Context & Success Criteria

- **What success looks like:** A model that can reliably identify pneumonia cases in pediatric chest X-rays with sensitivity and specificity both exceeding 95%, reducing radiologist workload while maintaining diagnostic accuracy. The model should integrate into clinical workflows without causing alert fatigue.

- **Failure mode costs:** 
  - False negative (missed pneumonia): Could lead to untreated patient, potential disease progression, complications, or worse clinical outcomes
  - False positive (false alarm): Will be discarded by physician review, but a too-high FP rate will cause user fatigue and reduce trust in the system

- **Non-negotiable constraints:**
  - Sensitivity > 95% (critical to not miss pneumonia cases)
  - Specificity > 95% (to avoid excessive false alarms)
  - Input: Anterior-posterior chest X-ray images (JPEG format)
  - Target population: Pediatric patients ages 1-5 years
  - **Interpretability:** Model must provide visual explanations (saliency maps/Grad-CAM) highlighting suspicious areas
  - **Confidence scores:** All predictions must include probability scores for clinical review
  - **Inference latency:** < 5 seconds per image for clinical workflow compatibility

- **Known domain pitfalls:**
  - Image quality variations despite QC screening
  - Overlapping radiographic features between pneumonia and other respiratory conditions
  - Potential bias from single-center data collection (Guangzhou Women and Children's Medical Center)
  - Age-related anatomical differences in pediatric lungs

## IV. Data Characterization

*Updated after EDA — 2026-03-26*

- **Total images:** 5,216 chest X-ray images (JPEG)
- **Classes:** 2 categories (Pneumonia: 3,875 / 74.3%, Normal: 1,341 / 25.7%)
- **Organization:** train/, test/, val/ folder structure
- **Source:** Retrospective cohorts from Guangzhou Women and Children's Medical Center, Guangzhou
- **Patient age range:** 1-5 years old
- **Image type:** Anterior-posterior chest radiographs
- **Quality control:** Low quality or unreadable scans removed
- **Labeling:** Diagnoses graded by two expert physicians, evaluation set checked by third expert
- **Inference requirements:**
  - Visual explanations (saliency maps/Grad-CAM) for all predictions
  - Confidence scores (probability 0-1) for each class
  - Inference latency < 5 seconds per image

### EDA Findings

- **Imbalance Status:** Moderate (2.9:1 ratio)
- **Imbalance Nature:** Enrichment artifact for dataset construction, not real-world prevalence
- **Validation Set:** Too small (16 images) — will use k-fold cross-validation with combined train+val
- **Baseline Target:** 80% accuracy (majority classifier achieves 74.3%)
- **Domain Expert Interpretations:**
  - Class imbalance is intentional for training, not reflective of real-world statistics
  - No specific pneumonia subtypes to prioritize (binary classification only)
  - K-fold cross-validation required due to small validation set

### Constraints on Architecture

- Class weights must be applied during training (inverse frequency weighting)
- K-fold cross-validation for reliable model evaluation
- Task requires deep learned features (baseline gap of 5.7% from majority classifier)
- No subtype differentiation — binary classification only

## V. Hypothesis History

| Version | Hypothesis | Experiment | Outcome | Domain Expert Sign-off |
| :--- | :--- | :--- | :--- | :--- |
| H-001 | Using a convolutional neural network architecture trained on 5,863 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection | — | Untested | Pending |
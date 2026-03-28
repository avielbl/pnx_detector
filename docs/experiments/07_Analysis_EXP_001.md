# Analysis: EXP-001

*Stage 7: Experiment Analysis*

---

### A. Experiment Overview

* **Experiment ID:** EXP-001
* **Run Type:** Baseline
* **Tracking Tool:** TensorBoard + CSVLogger — `tensorboard --logdir=logs`
* **Active Hypothesis:** "Using a convolutional neural network architecture trained on 5,863 quality-controlled pediatric chest X-ray images will achieve >95% sensitivity and specificity for pneumonia detection because the dataset provides sufficient diversity and expert-validated labels to learn discriminative features of pneumonia patterns in pediatric lungs."
* **TECHSPEC:** `docs/techspecs/TECHSPEC_EXP_001.md`

---

### B. TECHSPEC Evaluation (pre-committed criteria)

The TECHSPEC defined success tiers in terms of sensitivity and specificity (Section E). Evaluated against cross-validated mean across 5 folds:

| TECHSPEC Tier | Threshold | Achieved (mean ± SD) | Verdict |
| :--- | :--- | :--- | :--- |
| Best case | Sens ≥ 0.97 AND Spec ≥ 0.97 | 0.921 ± 0.010 / 0.979 ± 0.011 | NOT REACHED |
| Realistic | Sens ≥ 0.95 AND Spec ≥ 0.95 | same | NOT REACHED |
| **Worst case alive** | **Sens ≥ 0.90 OR Spec ≥ 0.90** | **both conditions met** | **REACHED** |
| Failure | Sens < 0.85 AND Spec < 0.85 | — | NOT REACHED |

* **Tier reached:** Worst case alive
* **Goalpost integrity:** Criteria evaluated exactly as committed in TECHSPEC. No post-hoc reinterpretation. The specificity target (0.95) is met; the sensitivity target (0.95) is not.
* **HPO eligible:** Yes — the model learns, converges cleanly, and is 3.5pp below the sensitivity target. Proceed to threshold tuning and/or transfer learning before declaring architecture failure.

**Per-fold breakdown:**

| Fold | val/f1 | Sensitivity | Specificity | Best Epoch |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.939 | 0.920 | 0.993 | 18 |
| 1 | 0.932 | 0.918 | 0.974 | 20 |
| 2 | 0.939 | 0.923 | 0.985 | 23 |
| 3 | 0.928 | 0.910 | 0.981 | 21 |
| 4 | 0.944 | 0.937 | 0.963 | 31 |
| **Mean ± SD** | **0.936 ± 0.006** | **0.921 ± 0.010** | **0.979 ± 0.011** | — |

---

### C. Hypothesis Verdict

* **Status:** INCONCLUSIVE
* **Evidence:** The model clearly learned discriminative features — mean F1 0.936 vs majority-class baseline 0.743, a +19.3pp improvement. Specificity 0.979 exceeds the 0.95 target. Sensitivity 0.921 falls 2.9pp short. The gap is consistent across all 5 folds (SD=0.010), indicating a systematic limit of the from-scratch training configuration rather than noise. The hypothesis is not falsified because the architecture is demonstrably learning; it is inconclusive because the primary performance target is not yet met.
* **Domain Expert Interpretation:** The sensitivity shortfall is to be improved, not accepted. False negatives (missed pneumonia) are the primary clinical risk. This result is a valid baseline to build on — threshold tuning and pretrained weights are the next steps before concluding anything about architecture viability.

---

### D. Requirement Verification

| Linked Requirement | PRD Target | Achieved (mean) | Status |
| :--- | :--- | :--- | :--- |
| `REQ-PERF-01` | Sensitivity > 0.95 | 0.921 ± 0.010 | **FAIL** |
| `REQ-PERF-02` | Specificity > 0.95 | 0.979 ± 0.011 | PASS |
| `REQ-PERF-03` | Low false positive rate | FPR ≈ 0.021 | PASS |
| `REQ-EXP-02` | Best model checkpoint saved | checkpoints/EXP-001/baseline/ | PASS |

REQ-SYS-01/02, REQ-DATA-01/02, REQ-EXP-01, REQ-INF-01/02/03 are infrastructure requirements verified in Stage 5 and not re-evaluated here.

---

### E. Failed Attempts — MANDATORY

| Approach / Config | Symptom | Root Cause | Lesson Learned |
| :--- | :--- | :--- | :--- |
| `AUROC(task="multiclass")` per step | val/auroc ≈ 0.021 in CSV | Lightning averages per-step scalars; single-class batches return 0 | Always accumulate probs in a buffer; compute rank metrics at epoch end only |
| `Recall(average="micro")` for sensitivity | val/sensitivity = val/accuracy | Micro-recall = accuracy for multiclass | Use `average=None` and index by class |
| `batch_size=32` on GTX 1650 | CUDA OOM | EfficientNet-B4 backprop peaks >4.3GB at bs=32 | Reduce to bs=16 (3.9GB stable); budget VRAM before choosing bs |
| `ModelCheckpoint(filename="...val/f1...")` + `glob` | checkpoint_path=None | `/` in metric name creates subdirectory; glob misses it | Use `rglob` for checkpoint discovery |
| CosineAnnealingWarmRestarts T_0=10 | Sensitivity spikes/dips at restart boundaries (e.g., fold 0 ep11: sens=0.690) | LR reset disrupts sensitivity-sensitive decision boundary | Consider OneCycleLR or ReduceLROnPlateau for next experiment; or increase T_0 |

---

### F. Error Analysis & Domain Cost

* **Primary failure mode:** False negatives — approximately 7.9% of PNEUMONIA cases misclassified as NORMAL at the default 0.5 threshold.
* **Domain cost:** Each false negative is a missed pneumonia in a pediatric patient (ages 1–5), risking untreated disease progression. The current FN rate is a baseline to improve, not an acceptable production threshold. With ~3,875 PNEUMONIA cases in the dataset, 7.9% FN ≈ 306 missed cases per training cycle. Clinical deployment requires sensitivity ≥ 0.95.
* **False positive rate:** FPR ≈ 2.1% (specificity 0.979) — low enough to avoid alert fatigue per PRD. This is already acceptable; optimizing further is not the priority.
* **Specificity variance note:** The wide per-fold specificity range (0.963–0.993) is likely a small-sample artifact. Each fold's validation set contains ~208 NORMAL images (25.7% of ~5,216/5 ≈ 1,043 val images). At this size, a single digit's change in true negatives shifts specificity by ~0.5pp. This observation motivated the data split decision below.

---

### G. Diagnostics & Insights

* **Convergence:** All 5 folds converge cleanly. No NaN/Inf in loss. Loss falls from ~0.43 at epoch 0 to ~0.16–0.22 at best epoch.
* **Sensitivity trajectory:** Sensitivity improves rapidly in early epochs (+5–7pp per epoch, ep0–6), then slows. The ceiling at ~0.921–0.937 is consistent with a threshold-limited problem: the model assigns high probability to PNEUMONIA but not high enough to clear 0.5 for borderline cases.
* **Scheduler behavior:** CosineAnnealingWarmRestarts with T_0=10 causes sensitivity dips at restart boundaries (notably fold 0 ep11: 0.820 → 0.690). The LR reset perturbs the sensitivity-critical decision boundary more than specificity. Specificity recovers quickly; sensitivity takes 2–3 epochs.
* **Fold 4 best performance:** Fold 4 trained to epoch 31 (vs. fold 0 early stop at 18), achieving sensitivity 0.937. This suggests more training epochs may not have been exhausted — the early stopping patience=10 on F1 may have terminated some folds prematurely while sensitivity was still improving.
* **Comparison to EDA:** EDA predicted 80% accuracy as baseline target. Achieved 93.6% F1 (equivalent to ~93–94% accuracy). Significantly exceeds the EDA baseline; the architecture is viable.

---

### H. Recommendations for Next Step

**Immediate action — data split revision (for all future experiments):**

The k-fold cross-validation was necessary because the original validation set was only 16 images. For EXP-002 onward, restructure the data split to **70% train / 20% val / 10% test** from the full 5,216-image pool. This yields:
- Train: ~3,651 images
- Val: ~1,043 images (sufficient for stable metric estimation)
- Test: ~522 images (held-out, never seen during training or threshold tuning)

Drop k-fold for subsequent experiments. Single train/val/test split with stable validation set is sufficient once the split is properly sized.

**EXP-002 candidates (both recommended, order TBD):**

**Option A — Threshold tuning (low effort, high expected sensitivity gain):**
- Keep the EXP-001 best checkpoint (fold 4, epoch 31, F1=0.944)
- Sweep decision threshold from 0.3–0.7 on the validation set
- Recompute sensitivity/specificity at each threshold; select operating point where sensitivity ≥ 0.95 with maximum specificity
- Expected: threshold ~0.35–0.40 should recover 3–4pp sensitivity at ~1–2pp specificity cost
- Risk: low. This is inference-only; no retraining required.

**Option B — Transfer learning (medium effort, likely larger gain):**
- Replace `pretrained=False` with `pretrained=True` (ImageNet weights) in EfficientNet-B4
- Use the 70/20/10 split above
- Fine-tune with lower learning rate (1e-5) and frozen backbone for first N epochs
- Expected: pretrained features capture low-level edge/texture patterns immediately, freeing capacity for pneumonia-specific patterns. Literature suggests 2–5pp sensitivity improvement over from-scratch baselines on medical imaging tasks of this size.
- Risk: medium. Requires retraining.

**Recommended order:** Option A first (validate the operating point can meet the target with existing model), then Option B (new experiment with full retraining if threshold tuning is insufficient or reduces specificity below 0.95).

---

### I. Clarification & Decision Log

* **Q1:** The active hypothesis appears inconclusive — the model learns but sensitivity falls short at 0.921. Falsification requires < 0.85; we are well above that. -> **User Decision:** Agreed, verdict is INCONCLUSIVE.
* **Q2:** At sensitivity 0.921, ~7.9% of pneumonia cases are missed. Given clinical failure cost (untreated pediatric pneumonia), how to characterize this? -> **User Decision:** Continue to improve it. Not a blocker; treat as a baseline to build on.
* **Q3:** Next experiment direction — threshold tuning vs. transfer learning? -> **User Decision:** Both. Run both as EXP-002 candidates.
* **Q4:** Specificity range 0.963–0.993 across folds — random artifact or signal? -> **User Decision:** Likely random due to small per-fold validation set size. Rebalance to 70/20/10 train/val/test split and drop k-fold for next experiments.

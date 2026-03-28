# Retrospective: EXP-001 — Stage 5 Infrastructure Review and Experiment Launch

**Session date:** 2026-03-27
**Covers:** Stage 5 (Infrastructure Build) gap review through Stage 6 (Experiment Execution) fold-0 launch

---

## What Happened

Reviewed Stage 5 infrastructure for gaps before proceeding to Stage 6. Found and fixed 9 bugs across the model, trainer, and environment before a single full training epoch ran. Fold 0 of EXP-001 was then launched successfully.

---

## Bugs Found and Fixed

### Stage 5 Pre-Flight (Code Review)

| # | File | Issue | Root Cause | Fix |
| :- | :--- | :--- | :--- | :--- |
| 1 | `efficientnet.py` | `AttributeError: 'PneumoniaModel' has no attribute 'test_predictions'` | `test_step` and `on_test_epoch_end` referenced buffers never initialized in `__init__` | Added `self.test_predictions = []` and `self.test_labels = []` to `__init__` |
| 2 | `efficientnet.py` | `val/sensitivity` = accuracy | `Recall(average="micro")` computes micro-recall = accuracy for multiclass, not per-class recall | Use fresh `Recall(average=None).to(device)` in epoch-end; index `[1]` for PNEUMONIA, `[0]` for NORMAL |
| 3 | `efficientnet.py` | `RuntimeError: tensors on different devices` | Fresh metric objects created on CPU while tensors were on CUDA | Added `.to(device)` on all fresh metric objects constructed in epoch-end hooks |

### Stage 6 Pre-Flight (GPU Probe Run)

| # | File | Issue | Root Cause | Fix |
| :- | :--- | :--- | :--- | :--- |
| 4 | `efficientnet.py` | `val/auroc ≈ 0.02` | Attempted fix used `AUROC(task="multiclass")` — different computation path for binary case | Switch to `AUROC(task="binary")` with `probs[:,1]` |
| 5 | `efficientnet.py` | `val/auroc ≈ 0.02` (root cause) | Lightning averages per-step scalar AUROC with `on_epoch=True`. Validation batches with single class (warned: "No positive samples") return 0; mean ≈ 0.021. Real AUROC confirmed 0.9648 via sklearn on checkpoint. | Removed per-step AUROC entirely. Accumulate `probs[:,1]` in `val_prob_positives` list; compute `AUROC(task="binary")` from full epoch data in `on_validation_epoch_end` |
| 6 | `trainer.py` | `UnicodeEncodeError` on Windows | Em-dash (`—`) and arrow (`→`) characters in print statements; Windows console uses cp1252 | Replaced all non-ASCII with ASCII equivalents (`--`, `->`) |
| 7 | `trainer.py` | `checkpoint_path=None`, `best_f1=0.0` | Lightning saves `best_epoch=00_val/f1=0.8204.ckpt` where `/` creates literal `val/` subdirectory; `glob("best_*.ckpt")` misses nested files | Use `rglob("*.ckpt")` excluding `last.ckpt`; robust float parser on filename parts |
| 8 | `pyproject.toml` | `ModuleNotFoundError: tensorboard` | TensorBoard not listed in dependencies | Added `tensorboard>=2.14` to `[project.dependencies]` |
| 9 | `pyproject.toml` | `torch 2.11.0+cpu`, CUDA unavailable | PyPI torch is CPU-only by default | Added `[[tool.uv.index]]` for pytorch-cu124 and `[tool.uv.sources]` routing torch+torchvision to CUDA index |

---

## Key Learnings

### TorchMetrics + Lightning: Per-Step Averaging Trap

The AUROC bug is non-obvious and likely to recur. When you call `self.log("val/auroc", auroc_metric(preds, probs), on_epoch=True)` in `validation_step`, Lightning accumulates the **scalar return value** as a running mean — not the internal state of the metric object. For rank-based metrics like AUROC, any batch that happens to contain only one class returns 0 (or warns and skips), and the mean collapses to near zero regardless of actual classification quality.

**Rule:** Never log AUROC, sensitivity, or specificity per-step with `on_epoch=True`. Always accumulate raw predictions/probabilities in Python lists and compute these metrics once in `on_validation_epoch_end` from the full epoch.

Stateful TorchMetrics objects (`Accuracy`, `F1Score`, `Precision`, `Recall`) are safe with `on_epoch=True` because Lightning calls `.compute()` on the metric's accumulated state at epoch end, not a mean of scalars. AUROC is fundamentally different.

### Binary AUROC in a 2-Class Setup

`AUROC(task="multiclass", num_classes=2)` computes a different quantity than `AUROC(task="binary")`. For 2-class problems, always use `AUROC(task="binary")` with `probs[:,1]` as the score. The multiclass variant averages per-class AUROCs using the OvR strategy, which produces misleading values for a true binary problem.

### Windows Encoding

Any Unicode character beyond ASCII (em-dashes, arrows, special symbols) in a print statement will crash on a Windows terminal using the default cp1252 encoding. Keep all console output pure ASCII in cross-platform code.

### Lightning Checkpoint Filenames with `/`

`ModelCheckpoint(filename="best_{epoch:02d}_{val/f1:.4f}")` — the `/` in `val/f1` creates a literal subdirectory `val/` inside the checkpoint directory. File discovery must use `rglob`, not `glob`. The subdirectory is stable and consistent; parsing just needs to account for it.

### GTX 1650 VRAM Budget

EfficientNet-B4 with optimizer state at batch_size=32: OOM on 4.3GB. At batch_size=16: 3.9GB peak (stable). Training speed: ~958s per 3 epochs = ~16 min/3 epochs = ~320s/epoch = ~4.4h/fold for 50 epochs.

---

## EXP-001 Fold 0 — Status at Session End

| Epoch | val/f1 | val/sensitivity | val/specificity | val/loss |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.835 | 0.785 | 0.978 | 0.426 |
| 1 | 0.840 | 0.792 | 0.978 | 0.380 |
| 2 | 0.861 | 0.816 | 0.989 | 0.338 |
| 3 | 0.879 | 0.842 | 0.985 | 0.335 |

GPU probe (separate 3-epoch run, confirmed loop correct): AUROC=0.9648 at epoch 2.

Trend: sensitivity improving +5.7pp/4 epochs, well above majority-class baseline (74.3%).

---

## What to Do Next Session

1. Check fold-0 final result and folds 1–4 completion
2. Proceed to Stage 7 (Analysis) once all 5 folds complete
3. Update `docs/experiments/06_Experiment_Log.md` with final per-fold results
4. Evaluate against TECHSPEC tiered success criteria (sensitivity ≥ 0.95, specificity ≥ 0.95)

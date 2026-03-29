"""EXP-002: Threshold tuning (Run A) and transfer learning (Run B).

Run A: Threshold sweep on EXP-001 best checkpoint. Inference-only, no retraining.
Run B: EfficientNet-B4 with ImageNet pretrained weights, fixed 70/20/10 split,
       early stopping on val/sensitivity.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pnx_detector.data import PneumoniaDataModule
from pnx_detector.models import PneumoniaModel
from pnx_detector.utils.threshold_optimizer import ThresholdOptimizer


# ---------------------------------------------------------------------------
# Run A: Threshold sweep
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    checkpoint_path: str,
    data_dir: str = "chest_xray_data",
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    thresholds: list = None,
) -> dict:
    """Sweep decision thresholds on val set using EXP-001 best checkpoint.

    Args:
        checkpoint_path: Path to the EXP-001 best checkpoint (.ckpt)
        data_dir: Path to data directory
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        seed: Random seed (must match training seed for reproducible split)
        thresholds: Thresholds to evaluate (default: [0.30, 0.35, 0.40, 0.45, 0.50])

    Returns:
        Dict with threshold sweep results and selected optimal threshold
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    print("\n" + "=" * 60)
    print("EXP-002 Run A: Threshold Sweep")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model from checkpoint
    model = PneumoniaModel.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
    model.eval()
    model.to(device)

    # Fixed split datamodule
    datamodule = PneumoniaDataModule(
        data_dir=data_dir,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        random_state=seed,
        split_mode="fixed",
    )
    datamodule.setup()

    # Collect all val probabilities and labels
    all_probs = []
    all_labels = []

    val_loader = datamodule.val_dataloader()
    print(f"Val set size: {len(val_loader.dataset)} images")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    prob_pos = all_probs[:, 1]

    # Evaluate each threshold
    optimizer = ThresholdOptimizer(target_sensitivity=0.95)
    print(f"\n{'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14} {'F1':<10}")
    print("-" * 52)

    results = []
    for t in thresholds:
        preds = (prob_pos >= t).astype(int)
        metrics = optimizer._calculate_metrics(preds, all_labels, prob_pos)
        results.append({"threshold": t, **metrics})
        marker = " <-- target" if metrics["sensitivity"] >= 0.95 and metrics["specificity"] >= 0.90 else ""
        print(f"{t:<12.2f} {metrics['sensitivity']:<14.4f} {metrics['specificity']:<14.4f} "
              f"{metrics['f1_score']:<10.4f}{marker}")

    # Also run the optimizer for the continuous optimal point
    opt_result = optimizer.optimize(prob_pos, all_labels)
    print(f"\nOptimizer best:  threshold={opt_result.optimal_threshold:.4f}  "
          f"sens={opt_result.sensitivity:.4f}  spec={opt_result.specificity:.4f}")

    # Save threshold curve plot
    plot_path = "docs/experiments/threshold_curve_exp002.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    optimizer.plot_threshold_curve(prob_pos, all_labels, save_path=plot_path)
    print(f"Threshold curve saved: {plot_path}")

    # Select best threshold: highest specificity where sensitivity >= 0.95
    meeting_target = [r for r in results if r["sensitivity"] >= 0.95]
    if meeting_target:
        best = max(meeting_target, key=lambda x: x["specificity"])
        print(f"\nSelected threshold: {best['threshold']:.2f}  "
              f"(sens={best['sensitivity']:.4f}, spec={best['specificity']:.4f})")
    else:
        best = opt_result.__dict__
        best["threshold"] = opt_result.optimal_threshold
        print(f"\nNo threshold in sweep met sens>=0.95. "
              f"Optimizer best: {opt_result.optimal_threshold:.4f}")

    print("=" * 60)
    return {"sweep": results, "selected": best, "optimizer_result": opt_result}


# ---------------------------------------------------------------------------
# Run B: Transfer learning
# ---------------------------------------------------------------------------

def run_transfer_learning(
    exp_id: str = "EXP-002",
    run_name: str = "transfer",
    data_dir: str = "chest_xray_data",
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    epochs: int = 50,
    frozen_epochs: int = 5,
    seed: int = 42,
    num_workers: int = 4,
) -> dict:
    """Run B: Train EfficientNet-B4 with ImageNet pretrained weights.

    Uses fixed 70/20/10 split. Freezes backbone for frozen_epochs, then
    unfreeze and fine-tune all layers. Early stopping on val/sensitivity.

    Args:
        exp_id: Experiment ID for logging
        run_name: Run name for checkpoint dir and logs
        data_dir: Path to data directory
        learning_rate: Learning rate (lower than EXP-001 for fine-tuning)
        batch_size: Batch size (constrained by GTX 1650 VRAM)
        epochs: Max epochs
        frozen_epochs: Epochs to keep backbone frozen (head-only training)
        seed: Random seed
        num_workers: DataLoader workers

    Returns:
        Dict with training results
    """
    import time
    from lightning import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    print("\n" + "=" * 60)
    print(f"EXP-002 Run B: Transfer Learning")
    print(f"  pretrained=True, lr={learning_rate}, frozen_epochs={frozen_epochs}")
    print("=" * 60)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: No GPU detected -- training will be slow on CPU.")

    # Fixed split datamodule
    datamodule = PneumoniaDataModule(
        data_dir=data_dir,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        random_state=seed,
        split_mode="fixed",
    )

    class_weights = datamodule.get_class_weights()

    # Phase 1: frozen backbone (head only)
    print(f"\nPhase 1: Training head only for {frozen_epochs} epochs (backbone frozen)...")
    model = PneumoniaModel(
        num_classes=2,
        learning_rate=learning_rate,
        weight_decay=1e-2,
        dropout=0.3,
        class_weights=class_weights,
        pretrained=True,
    )

    # Freeze all backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    checkpoint_dir = Path("checkpoints") / exp_id / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir="logs", name=f"{exp_id}_{run_name}", version="phase1")
    csv_logger = CSVLogger(save_dir="logs", name=f"{exp_id}_{run_name}", version="phase1")

    trainer_phase1 = Trainer(
        max_epochs=frozen_epochs,
        accelerator="auto",
        devices=1,
        logger=[tb_logger, csv_logger],
        enable_progress_bar=True,
        enable_checkpointing=False,
        precision="32-true",
        log_every_n_steps=10,
    )

    start = time.time()
    trainer_phase1.fit(model, datamodule)
    print(f"Phase 1 complete ({time.time() - start:.0f}s)")

    # Phase 2: unfreeze all, fine-tune with early stopping on val/sensitivity
    print(f"\nPhase 2: Fine-tuning all layers (max {epochs} epochs, "
          f"early stop on val/sensitivity)...")
    for param in model.backbone.parameters():
        param.requires_grad = True

    callbacks = [
        EarlyStopping(
            monitor="val/sensitivity",
            mode="max",
            patience=10,
            min_delta=0.001,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_{epoch:02d}_{val/sensitivity:.4f}",
            monitor="val/sensitivity",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    tb_logger2 = TensorBoardLogger(save_dir="logs", name=f"{exp_id}_{run_name}", version="phase2")
    csv_logger2 = CSVLogger(save_dir="logs", name=f"{exp_id}_{run_name}", version="phase2")
    print(f"TensorBoard: tensorboard --logdir=logs")

    trainer_phase2 = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=[tb_logger2, csv_logger2],
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="checkpoints",
        precision="32-true",
        log_every_n_steps=10,
    )

    start = time.time()
    trainer_phase2.fit(model, datamodule)
    training_time = time.time() - start

    # Find best checkpoint
    checkpoints = [p for p in checkpoint_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
    best_checkpoint = None
    best_sens = 0.0
    if checkpoints:
        def _parse_sens(p: Path) -> float:
            for part in p.name.replace(".ckpt", "").split("="):
                try:
                    return float(part)
                except ValueError:
                    pass
            return 0.0
        best_checkpoint = max(checkpoints, key=_parse_sens)
        best_sens = _parse_sens(best_checkpoint)

    final_metrics = trainer_phase2.callback_metrics

    results = {
        "exp_id": exp_id,
        "run_name": run_name,
        "val_f1": final_metrics.get("val/f1", torch.tensor(0)).item(),
        "val_sensitivity": final_metrics.get("val/sensitivity", torch.tensor(0)).item(),
        "val_specificity": final_metrics.get("val/specificity", torch.tensor(0)).item(),
        "val_loss": final_metrics.get("val/loss", torch.tensor(-1)).item(),
        "best_sensitivity": best_sens,
        "checkpoint_path": str(best_checkpoint) if best_checkpoint else None,
        "training_time_seconds": training_time,
    }

    print(f"\n{'=' * 60}")
    print(f"Run B Complete")
    print(f"  val/sensitivity: {results['val_sensitivity']:.4f}")
    print(f"  val/specificity: {results['val_specificity']:.4f}")
    print(f"  val/f1:          {results['val_f1']:.4f}")
    print(f"  Best checkpoint: {results['checkpoint_path']}")
    print(f"  Training time:   {training_time:.0f}s ({training_time/3600:.2f}h)")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EXP-002: Threshold sweep and transfer learning")
    parser.add_argument("--run", choices=["a", "b", "both"], default="both",
                        help="Which run to execute (default: both)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/EXP-001/baseline/best_epoch=31_val/f1=0.9436.ckpt",
                        help="EXP-001 checkpoint for Run A threshold sweep")
    parser.add_argument("--data-dir", type=str, default="chest_xray_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    results = {}

    if args.run in ("a", "both"):
        results["run_a"] = run_threshold_sweep(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    if args.run in ("b", "both"):
        results["run_b"] = run_transfer_learning(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    return results


if __name__ == "__main__":
    main()

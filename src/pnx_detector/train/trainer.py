"""Training script for pneumonia detection experiment EXP-001.

Implements Stage 6: Training experiment according to TECHSPEC_EXP_001.md.
Runs EfficientNet-B4 from scratch with heavy augmentation and class weights.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pnx_detector.data import PneumoniaDataModule, get_train_transforms, get_val_transforms
from pnx_detector.models import PneumoniaModel
from pnx_detector.tracking import init_clearml_task, make_clearml_logger


def create_callbacks(exp_id: str, run_name: str, best_f1: float = 0.0) -> list:
    """Create Lightning callbacks for training.

    Args:
        exp_id: Experiment identifier
        run_name: Name of this run
        best_f1: Best F1 score so far (for resuming)

    Returns:
        List of Lightning callbacks
    """
    callbacks = []

    # Early stopping based on validation F1
    callbacks.append(
        EarlyStopping(
            monitor="val/f1",
            mode="max",
            patience=10,
            verbose=True,
            min_delta=0.001,
        )
    )

    # Model checkpointing - save best checkpoint by F1
    checkpoint_dir = Path("checkpoints") / exp_id / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_{epoch:02d}_{val/f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
    )

    return callbacks


def run_training_experiment(
    exp_id: str = "EXP-001",
    run_name: str = "baseline_fold0",
    data_dir: str = "chest_xray_data",
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 50,
    num_folds: int = 5,
    current_fold: int = 0,
    seed: int = 42,
    num_workers: int = 4,
    max_gpu_hours: float = 8.0,
) -> Dict[str, Any]:
    """Run a single training fold for the experiment.

    Args:
        exp_id: Experiment identifier
        run_name: Name of this run
        data_dir: Path to the data directory
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        num_folds: Total number of folds
        current_fold: Current fold index (0 to num_folds-1)
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        max_gpu_hours: Maximum GPU hours for this run

    Returns:
        Dictionary with training results
    """
    # Set random seeds
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # GPU validation - fail loudly if CUDA is expected but unavailable
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No CUDA GPU detected - training will run on CPU and be significantly slower.")
        print("         If you expected a GPU, check your CUDA installation and driver.")

    # Initialize ClearML task (optional - skip if not configured)
    try:
        task = init_clearml_task(
            project_name="pnx_detector",
            task_name=f"{exp_id}_{run_name}_fold{current_fold}",
            hyperparams={
                "exp_id": exp_id,
                "run_name": run_name,
                "fold": current_fold,
                "num_folds": num_folds,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
                "num_workers": num_workers,
                "max_gpu_hours": max_gpu_hours,
                "architecture": "efficientnet-b4",
                "augmentation": "heavy",
                "optimizer": "AdamW",
                "lr_scheduler": "CosineAnnealingWarmRestarts",
            },
            tags=[exp_id, "baseline", f"fold-{current_fold}"],
        )
        clearml_available = True
    except Exception as e:
        print(f"Warning: ClearML not configured, skipping experiment tracking: {e}")
        task = None
        clearml_available = False

    print(f"\n{'='*60}")
    print(f"Starting training: {exp_id}_{run_name} (fold {current_fold}/{num_folds})")
    print(f"{'='*60}\n")

    # Create data module
    datamodule = PneumoniaDataModule(
        data_dir=data_dir,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        num_splits=num_folds,
        random_state=seed,
    )

    # Get class weights
    class_weights = datamodule.get_class_weights()

    # Create model
    model = PneumoniaModel(
        num_classes=2,
        learning_rate=learning_rate,
        weight_decay=1e-2,
        dropout=0.3,
        class_weights=class_weights,
    )

    # Set fold for cross-validation
    datamodule.set_fold(current_fold)

    # Create callbacks
    callbacks = create_callbacks(exp_id, run_name)

    # TensorBoard logger - one version per fold so runs don't overwrite each other
    from lightning.pytorch.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{exp_id}_{run_name}",
        version=f"fold_{current_fold}",
    )
    csv_logger = CSVLogger(save_dir="logs", name=f"{exp_id}_{run_name}", version=f"fold_{current_fold}")
    logger = [tb_logger, csv_logger]
    print(f"TensorBoard logs: logs/{exp_id}_{run_name}/fold_{current_fold}/")
    print(f"  -> Run: tensorboard --logdir=logs")

    # Create trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="checkpoints",
        precision="32-true",
        log_every_n_steps=10,
    )

    # Track training start time
    import time
    start_time = time.time()

    # Train the model
    trainer.fit(model, datamodule)

    # Calculate training time
    training_time = time.time() - start_time
    gpu_hours = training_time / 3600  # Approximate GPU hours

    # Get best checkpoint
    checkpoint_path = None
    best_f1 = 0.0
    best_epoch = 0

    checkpoint_dir = Path("checkpoints") / exp_id / run_name
    if checkpoint_dir.exists():
        # rglob because Lightning creates val/f1=*.ckpt in a subdirectory when metric has /
        checkpoints = [p for p in checkpoint_dir.rglob("*.ckpt") if p.name != "last.ckpt"]
        if checkpoints:
            # Parse F1 from filename stem (e.g. "f1=0.8204")
            def _parse_f1(p: Path) -> float:
                for part in p.name.replace(".ckpt", "").split("="):
                    try:
                        return float(part)
                    except ValueError:
                        pass
                return 0.0

            best_checkpoint = max(checkpoints, key=_parse_f1)
            checkpoint_path = str(best_checkpoint)
            best_f1 = _parse_f1(best_checkpoint)
            # epoch from parent dirs or filename (e.g. best_epoch=02_val/f1=0.86.ckpt)
            for part in best_checkpoint.parts:
                if "epoch=" in part:
                    try:
                        best_epoch = int(part.split("epoch=")[-1].split("_")[0])
                    except ValueError:
                        pass

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = trainer.validate(model, datamodule)

    # Get final metrics
    final_metrics = trainer.callback_metrics

    # Calculate total GPU hours for all folds
    total_gpu_hours = gpu_hours * num_folds

    # Log final results to ClearML (if available)
    clearml_url = "N/A (ClearML not configured)"
    if clearml_available and task is not None:
        try:
            task.get_logger().report_scalar(
                title="final/val_loss",
                series="final/val_loss",
                value=final_metrics.get("val/loss", -1).item(),
                iteration=epochs,
            )
            task.get_logger().report_scalar(
                title="final/val_f1",
                series="final/val_f1",
                value=final_metrics.get("val/f1", 0).item(),
                iteration=epochs,
            )
            task.get_logger().report_scalar(
                title="final/val_sensitivity",
                series="final/val_sensitivity",
                value=final_metrics.get("val/sensitivity", 0).item(),
                iteration=epochs,
            )
            task.get_logger().report_scalar(
                title="final/val_specificity",
                series="final/val_specificity",
                value=final_metrics.get("val/specificity", 0).item(),
                iteration=epochs,
            )

            # Upload checkpoint
            if checkpoint_path:
                task.upload_artifact(
                    name="best_checkpoint",
                    artifact_object=checkpoint_path,
                )

            clearml_url = task.get_output_log_web_page()
        except Exception as e:
            print(f"Warning: Failed to log to ClearML: {e}")

    # Create results dictionary
    results = {
        "exp_id": exp_id,
        "run_name": run_name,
        "fold": current_fold,
        "num_folds": num_folds,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "val_loss": final_metrics.get("val/loss", -1).item(),
        "val_f1": final_metrics.get("val/f1", 0).item(),
        "val_sensitivity": final_metrics.get("val/sensitivity", 0).item(),
        "val_specificity": final_metrics.get("val/specificity", 0).item(),
        "val_accuracy": final_metrics.get("val/acc", 0).item(),
        "training_time_seconds": training_time,
        "gpu_hours": gpu_hours,
        "total_gpu_hours": total_gpu_hours,
        "checkpoint_path": checkpoint_path,
        "clearml_url": clearml_url,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Complete: {exp_id}_{run_name} (fold {current_fold}/{num_folds})")
    print(f"{'='*60}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Final Val Loss: {results['val_loss']:.4f}")
    print(f"Final Val Sensitivity: {results['val_sensitivity']:.4f}")
    print(f"Final Val Specificity: {results['val_specificity']:.4f}")
    print(f"Training Time: {training_time:.2f}s ({gpu_hours:.2f} GPU hours)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ClearML URL: {results['clearml_url']}")
    print(f"{'='*60}\n")

    return results


def run_all_folds(
    exp_id: str = "EXP-001",
    run_name: str = "baseline",
    data_dir: str = "chest_xray_data",
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 50,
    num_folds: int = 5,
    seed: int = 42,
    num_workers: int = 4,
    max_gpu_hours: float = 8.0,
) -> list:
    """Run all k-fold training experiments.

    Args:
        exp_id: Experiment identifier
        run_name: Base name for all runs
        data_dir: Path to the data directory
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        num_folds: Total number of folds
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        max_gpu_hours: Maximum GPU hours per fold

    Returns:
        List of results dictionaries
    """
    all_results = []

    for fold in range(num_folds):
        print(f"\n{'#'*60}")
        print(f"Starting Fold {fold}/{num_folds}")
        print(f"{'#'*60}\n")

        try:
            result = run_training_experiment(
                exp_id=exp_id,
                run_name=run_name,
                data_dir=data_dir,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                num_folds=num_folds,
                current_fold=fold,
                seed=seed,
                num_workers=num_workers,
                max_gpu_hours=max_gpu_hours,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nFAILED: Fold {fold} failed with error: {e}")
            all_results.append({
                "exp_id": exp_id,
                "run_name": run_name,
                "fold": fold,
                "error": str(e),
                "status": "failed",
            })

    return all_results


def main():
    """Main entry point for training."""
    import numpy as np

    # Configuration from TECHSPEC_EXP_001
    config = {
        "exp_id": "EXP-001",
        "run_name": "baseline",
        "data_dir": "chest_xray_data",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "num_folds": 5,
        "seed": 42,
        "num_workers": 4,
        "max_gpu_hours": 8.0,
    }

    print("\n" + "="*60)
    print("PNX Detector - Training Experiment EXP-001")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Run all folds
    results = run_all_folds(**config)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nSuccessful folds: {len(successful)}/{len(results)}")
    print(f"Failed folds: {len(failed)}/{len(results)}")

    if successful:
        best_result = max(successful, key=lambda x: x.get("val_f1", 0))
        print(f"\nBest fold: {best_result['fold']}")
        print(f"Best F1: {best_result['val_f1']:.4f}")
        print(f"Best Sensitivity: {best_result['val_sensitivity']:.4f}")
        print(f"Best Specificity: {best_result['val_specificity']:.4f}")
        print(f"Best checkpoint: {best_result['checkpoint_path']}")
        print(f"ClearML URL: {best_result['clearml_url']}")

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    main()
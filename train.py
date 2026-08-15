"""
train.py - Reproducible Training Pipeline for KLA Image Restoration

Connects the verified components:
    src/dataset.py  ->  DataLoader
    src/model.py    ->  KLARestorationModel (630K params)
    src/losses.py   ->  RestorationL1Loss

Usage:
    Smoke test (quick verification):
        python train.py --train-zip train.zip --smoke-test

    Full training:
        python train.py --train-zip train.zip --epochs 100

    Custom configuration:
        python train.py --train-zip train.zip --epochs 50 --batch-size 16 --lr 2e-4
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Project imports
from src.dataset import create_train_val_datasets
from src.model import KLARestorationModel, count_parameters
from src.losses import RestorationL1Loss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[train] Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    """Select CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[train] Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print(f"[train] Device: {device} (CUDA not available)")
    return device


# ---------------------------------------------------------------------------
# Training for one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The restoration model.
    dataloader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function (RestorationL1Loss).
    optimizer : torch.optim.Optimizer
        AdamW optimizer.
    device : torch.device
        Training device (cuda or cpu).
    epoch : int
        Current epoch number (for logging).
    use_amp : bool
        Whether to use automatic mixed precision.
    scaler : torch.amp.GradScaler or None
        AMP gradient scaler (required if use_amp is True).

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        noisy = batch["noisy"].to(device)    # [B, 1, 128, 128]
        gt = batch["gt"].to(device)          # [B, 1, 256, 256]

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                prediction = model(noisy)    # [B, 1, 256, 256]
                loss = criterion(prediction, gt)
            # Check for NaN/Inf before scaling
            _check_loss(loss, epoch, batch_idx)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            prediction = model(noisy)        # [B, 1, 256, 256]
            loss = criterion(prediction, gt)
            # Check for NaN/Inf
            _check_loss(loss, epoch, batch_idx)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Evaluate the model on the validation set.

    Parameters
    ----------
    model : nn.Module
        The restoration model.
    dataloader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function (RestorationL1Loss).
    device : torch.device
        Device.
    use_amp : bool
        Whether to use automatic mixed precision.

    Returns
    -------
    float
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        noisy = batch["noisy"].to(device)    # [B, 1, 128, 128]
        gt = batch["gt"].to(device)          # [B, 1, 256, 256]

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                prediction = model(noisy)
                loss = criterion(prediction, gt)
        else:
            prediction = model(noisy)
            loss = criterion(prediction, gt)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------

def _check_loss(loss: torch.Tensor, epoch: int, batch_idx: int) -> None:
    """Raise an error if loss contains NaN or Inf."""
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(
            f"[train] NaN/Inf loss detected at epoch {epoch}, "
            f"batch {batch_idx}. Loss value: {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: Dict[str, Any],
) -> None:
    """Save a training checkpoint.

    Saved keys:
        model_state_dict, optimizer_state_dict, epoch,
        train_loss, val_loss, config
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"[train] Checkpoint saved to {path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration - Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--train-zip", type=str, required=True,
        help="Path to train.zip containing NoisyLR/ and GT/ folders.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Checkpointing
    parser.add_argument(
        "--checkpoint-path", type=str, default="weights/best_model.pth",
        help="Path to save the best model checkpoint.",
    )

    # Mixed precision
    parser.add_argument(
        "--amp", action="store_true", default=False,
        help="Enable automatic mixed precision (AMP).",
    )

    # Smoke test
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help="Run a quick 1-epoch smoke test with a small data subset.",
    )

    # DataLoader
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader workers.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    print("=" * 64)
    print("KLA AI-Based Image Restoration - Training")
    print("=" * 64)

    # ---- Configuration summary -------------------------------------------
    config: Dict[str, Any] = {
        "train_zip": args.train_zip,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": args.amp,
        "smoke_test": args.smoke_test,
        "num_workers": args.num_workers,
        "checkpoint_path": args.checkpoint_path,
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # ---- Reproducibility -------------------------------------------------
    set_seed(args.seed)

    # ---- Device ----------------------------------------------------------
    device = select_device()

    # ---- Data loading ----------------------------------------------------
    print(f"\n[train] Loading dataset from {args.train_zip} ...")
    train_dataset, val_dataset = create_train_val_datasets(
        data_source=args.train_zip,
        seed=args.seed,
    )
    print(f"[train] Full dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # ---- Smoke test subsetting -------------------------------------------
    if args.smoke_test:
        smoke_train_size = min(32, len(train_dataset))
        smoke_val_size = min(8, len(val_dataset))
        train_dataset = Subset(train_dataset, list(range(smoke_train_size)))
        val_dataset = Subset(val_dataset, list(range(smoke_val_size)))
        args.epochs = 1
        print(
            f"[train] SMOKE TEST: using {len(train_dataset)} train, "
            f"{len(val_dataset)} val, 1 epoch"
        )

    # ---- DataLoaders -----------------------------------------------------
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(
        f"[train] DataLoaders: "
        f"{len(train_loader)} train batches, "
        f"{len(val_loader)} val batches "
        f"(batch_size={args.batch_size})"
    )

    # ---- Model -----------------------------------------------------------
    model = KLARestorationModel()
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"[train] Model: KLARestorationModel ({num_params:,} parameters)")

    # ---- Loss & Optimizer ------------------------------------------------
    criterion = RestorationL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"[train] Loss: RestorationL1Loss")
    print(f"[train] Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")

    # ---- AMP setup -------------------------------------------------------
    scaler: Optional[torch.amp.GradScaler] = None
    if args.amp:
        if device.type == "cuda":
            scaler = torch.amp.GradScaler(device="cuda")
            print("[train] AMP enabled (float16 on CUDA)")
        else:
            print("[train] AMP requested but CUDA not available; disabled")
            args.amp = False

    # ---- Training loop ---------------------------------------------------
    best_val_loss = float("inf")
    checkpoint_path = Path(args.checkpoint_path)

    print(f"\n{'='*64}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time (s)':>10}  {'Status':>8}")
    print(f"{'-'*64}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=args.amp,
            scaler=scaler,
        )

        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
        )

        epoch_time = time.time() - epoch_start

        # Check for best model
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
            )
            status = "* best"

        print(
            f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
            f"{epoch_time:>10.2f}  {status:>8}"
        )

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"Training complete.")
    print(f"  Total epochs:       {args.epochs}")
    print(f"  Best val loss:      {best_val_loss:.6f}")
    print(f"  Checkpoint:         {checkpoint_path}")

    if args.smoke_test:
        # Additional verification for smoke test
        assert checkpoint_path.is_file(), (
            f"Smoke test failed: checkpoint not saved at {checkpoint_path}"
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt, "Checkpoint missing model_state_dict"
        assert "optimizer_state_dict" in ckpt, "Checkpoint missing optimizer_state_dict"
        assert "epoch" in ckpt, "Checkpoint missing epoch"
        assert "val_loss" in ckpt, "Checkpoint missing val_loss"
        print(f"\n[OK] Smoke test PASSED")
        print(f"  - Data loading:     OK")
        print(f"  - Forward pass:     OK")
        print(f"  - Loss computation: OK")
        print(f"  - Backward pass:    OK")
        print(f"  - Optimizer step:   OK")
        print(f"  - Validation:       OK")
        print(f"  - Checkpoint save:  OK ({checkpoint_path})")

    print("=" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

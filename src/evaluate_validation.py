"""
src/evaluate_validation.py - Validation-Only Evaluation for KLA Image Restoration

Loads the trained best_model.pth checkpoint, runs inference on the 320
validation images (from the deterministic seed=42 split), computes
MSE / PSNR / SSIM per image, and saves:
    - results/validation_metrics.csv   (per-image + aggregate metrics)
    - results/validation_samples/      (visual comparisons for 5 samples)

Usage:
    python -m src.evaluate_validation \\
        --checkpoint weights/best_model.pth \\
        --train-zip train.zip

    Or with an explicit results directory:
    python -m src.evaluate_validation \\
        --checkpoint weights/best_model.pth \\
        --train-zip train.zip \\
        --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.dataset import create_train_val_datasets
from src.model import KLARestorationModel, count_parameters
from src.metrics import mse, psnr, ssim


# ---------------------------------------------------------------------------
# Visual comparison helper
# ---------------------------------------------------------------------------

def _save_comparison(
    noisy_np: np.ndarray,
    restored_np: np.ndarray,
    gt_np: np.ndarray,
    sample_id: str,
    save_path: Path,
) -> None:
    """Save a side-by-side comparison image: Input | Restored | GT.

    Parameters
    ----------
    noisy_np : np.ndarray
        Input NoisyLR image (128x128), shown upscaled for visual alignment.
    restored_np : np.ndarray
        Model output (256x256), clamped to [0, 1].
    gt_np : np.ndarray
        Ground truth (256x256), values in [0, 1].
    sample_id : str
        Sample filename stem for the title.
    save_path : Path
        Where to save the comparison figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[eval] matplotlib not installed; skipping visual for {sample_id}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(noisy_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Input NoisyLR\n(128x128)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(restored_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Restored\n(256x256)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Ground Truth\n(256x256)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(f"Sample: {sample_id}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration - Validation Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pth checkpoint.",
    )
    parser.add_argument(
        "--train-zip", type=str, required=True,
        help="Path to train.zip (used to reconstruct the seed=42 val split).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split (must match training).",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save metrics CSV and visual comparisons.",
    )
    parser.add_argument(
        "--num-visual", type=int, default=5,
        help="Number of visual comparison samples to save.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (1 for per-sample timing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("KLA Image Restoration - Validation Evaluation")
    print("=" * 64)

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # ---- Load checkpoint -------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[eval] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[eval]   Saved at epoch:    {ckpt.get('epoch', 'N/A')}")
    print(f"[eval]   Saved train loss:  {ckpt.get('train_loss', 'N/A')}")
    print(f"[eval]   Saved val loss:    {ckpt.get('val_loss', 'N/A')}")

    # ---- Model -----------------------------------------------------------
    model = KLARestorationModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[eval] Model loaded: {count_parameters(model):,} parameters")

    # ---- Validation dataset ----------------------------------------------
    print(f"\n[eval] Loading validation split from {args.train_zip} (seed={args.seed})")
    _, val_dataset = create_train_val_datasets(
        data_source=args.train_zip,
        seed=args.seed,
    )
    num_val = len(val_dataset)
    print(f"[eval] Validation samples: {num_val}")

    # ---- Output directories ----------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = results_dir / "validation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "validation_metrics.csv"

    # ---- Inference and metrics -------------------------------------------
    print(f"\n[eval] Running inference on {num_val} validation images ...")

    all_mse: List[float] = []
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_times: List[float] = []
    rows: List[Dict[str, Any]] = []

    # Select indices for visual comparisons (evenly spaced)
    num_visual = min(args.num_visual, num_val)
    visual_indices = set(
        np.linspace(0, num_val - 1, num_visual, dtype=int).tolist()
    )

    with torch.no_grad():
        for idx in range(num_val):
            sample = val_dataset[idx]
            noisy = sample["noisy"].unsqueeze(0).to(device)  # [1, 1, 128, 128]
            gt = sample["gt"]                                # [1, 256, 256]
            sample_id = sample["sample_id"]

            # Timed inference
            t_start = time.time()
            prediction = model(noisy)                        # [1, 1, 256, 256]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            inference_time = t_end - t_start
            all_times.append(inference_time)

            # Move prediction to CPU for metrics
            pred_cpu = prediction.squeeze(0).detach().cpu()  # [1, 256, 256]

            # Compute metrics (PSNR and SSIM clamp prediction to [0,1])
            val_mse = mse(pred_cpu, gt)
            val_psnr = psnr(pred_cpu, gt)
            val_ssim = ssim(pred_cpu, gt)

            all_mse.append(val_mse)
            all_psnr.append(val_psnr)
            all_ssim.append(val_ssim)

            rows.append({
                "sample_id": sample_id,
                "mse": val_mse,
                "psnr": val_psnr,
                "ssim": val_ssim,
                "inference_time_s": inference_time,
            })

            # Save visual comparison for selected samples
            if idx in visual_indices:
                # Prepare numpy arrays for plotting
                noisy_np = sample["noisy"].squeeze(0).numpy()        # [128, 128]
                pred_np = pred_cpu.squeeze(0).numpy()                # [256, 256]
                pred_np_clipped = np.clip(pred_np, 0.0, 1.0)
                gt_np = gt.squeeze(0).numpy()                        # [256, 256]

                save_name = samples_dir / f"comparison_{sample_id}.png"
                _save_comparison(
                    noisy_np=np.clip(noisy_np, 0.0, 1.0),
                    restored_np=pred_np_clipped,
                    gt_np=gt_np,
                    sample_id=sample_id,
                    save_path=save_name,
                )

            # Progress
            if (idx + 1) % 50 == 0 or (idx + 1) == num_val:
                print(f"  [{idx + 1}/{num_val}] PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")

    # ---- Aggregate -------------------------------------------------------
    mean_mse = float(np.mean(all_mse))
    mean_psnr = float(np.mean(all_psnr))
    mean_ssim = float(np.mean(all_ssim))
    total_time = float(np.sum(all_times))
    avg_time = float(np.mean(all_times))

    # ---- Save CSV --------------------------------------------------------
    fieldnames = ["sample_id", "mse", "psnr", "ssim", "inference_time_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        # Write aggregate row
        writer.writerow({
            "sample_id": "MEAN",
            "mse": mean_mse,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "inference_time_s": avg_time,
        })
    print(f"\n[eval] Metrics saved to {csv_path}")

    # ---- Print summary ---------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"Validation Evaluation Summary")
    print(f"{'=' * 64}")
    print(f"  Checkpoint:          {ckpt_path}")
    print(f"  Checkpoint epoch:    {ckpt.get('epoch', 'N/A')}")
    print(f"  Validation samples:  {num_val}")
    print(f"  Device:              {device}")
    print(f"{'  ':-<62}")
    print(f"  Mean MSE:            {mean_mse:.6f}")
    print(f"  Mean PSNR:           {mean_psnr:.2f} dB")
    print(f"  Mean SSIM:           {mean_ssim:.6f}")
    print(f"{'  ':-<62}")
    print(f"  Total inference:     {total_time:.2f} s")
    print(f"  Avg per sample:      {avg_time * 1000:.2f} ms")
    print(f"{'  ':-<62}")
    print(f"  CSV:                 {csv_path}")
    print(f"  Visual samples:      {samples_dir}/ ({num_visual} images)")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

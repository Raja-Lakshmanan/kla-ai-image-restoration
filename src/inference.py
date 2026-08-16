"""
src/inference.py - Final Test Inference Pipeline for KLA Image Restoration

Loads the trained KLARestorationModel checkpoint and runs inference on
the 400 test NoisyLR images, saving restored outputs as .npy files.

Usage:
    Quick test (3 images):
        python -m src.inference \
            --checkpoint weights/best_model.pth \
            --test-zip Test_NoisyLR.zip \
            --output-dir results/test_outputs \
            --limit 3

    Full inference (400 images):
        python -m src.inference \
            --checkpoint weights/best_model.pth \
            --test-zip Test_NoisyLR.zip \
            --output-dir results/test_outputs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.dataset import create_test_dataset
from src.model import KLARestorationModel, count_parameters


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration - Test Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--test-zip", type=str, required=True,
        help="Path to Test_NoisyLR.zip or extracted test directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save restored .npy outputs.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N test images (for quick testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def _validate_output(arr: np.ndarray, sample_id: str) -> None:
    """Validate a single restored output array before saving."""
    if arr.shape != (256, 256):
        raise ValueError(
            f"Output '{sample_id}' has shape {arr.shape}, expected (256, 256)"
        )
    if arr.dtype != np.float32:
        raise ValueError(
            f"Output '{sample_id}' has dtype {arr.dtype}, expected float32"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Output '{sample_id}' contains NaN or Inf values"
        )
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"Output '{sample_id}' has values outside [0, 1]: "
            f"min={arr.min():.6f}, max={arr.max():.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("KLA Image Restoration - Test Inference")
    print("=" * 64)

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")

    # ---- Load checkpoint -------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[inference] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[inference]   Saved at epoch:   {ckpt.get('epoch', 'N/A')}")
    print(f"[inference]   Saved val loss:   {ckpt.get('val_loss', 'N/A')}")

    # ---- Model -----------------------------------------------------------
    model = KLARestorationModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[inference] Model loaded: {count_parameters(model):,} parameters")

    # ---- Test dataset ----------------------------------------------------
    print(f"\n[inference] Loading test data from {args.test_zip}")
    test_dataset = create_test_dataset(data_source=args.test_zip)
    total_samples = len(test_dataset)
    print(f"[inference] Test samples: {total_samples}")

    # Apply --limit if specified
    num_to_process = total_samples
    if args.limit is not None:
        num_to_process = min(args.limit, total_samples)
        print(f"[inference] --limit={args.limit}: processing {num_to_process} images")

    # ---- Output directory ------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inference] Output directory: {output_dir}")

    # ---- Inference -------------------------------------------------------
    print(f"\n[inference] Running inference on {num_to_process} test images ...")

    all_times: list[float] = []
    saved_count = 0

    with torch.no_grad():
        for idx in range(num_to_process):
            sample = test_dataset[idx]
            noisy = sample["noisy"].unsqueeze(0).to(device)  # [1, 1, 128, 128]
            sample_id = sample["sample_id"]

            # Timed inference
            t_start = time.time()
            prediction = model(noisy)                        # [1, 1, 256, 256]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()

            inference_time = t_end - t_start
            all_times.append(inference_time)

            # Post-process: CPU → float32 → clamp [0,1] → remove dims
            restored = prediction.detach().cpu().float()     # [1, 1, 256, 256]
            restored = torch.clamp(restored, 0.0, 1.0)
            restored_np = restored.squeeze(0).squeeze(0).numpy()  # (256, 256)

            # Validate before saving
            _validate_output(restored_np, sample_id)

            # Save as .npy preserving the original sample ID
            save_path = output_dir / f"{sample_id}.npy"
            np.save(save_path, restored_np)
            saved_count += 1

            # Progress
            if (idx + 1) % 50 == 0 or (idx + 1) == num_to_process:
                print(
                    f"  [{idx + 1}/{num_to_process}] "
                    f"Saved {sample_id}.npy  "
                    f"({inference_time * 1000:.1f} ms)"
                )

    # ---- Summary ---------------------------------------------------------
    total_time = sum(all_times)
    avg_time = total_time / max(saved_count, 1)

    print(f"\n{'=' * 64}")
    print(f"Test Inference Summary")
    print(f"{'=' * 64}")
    print(f"  Checkpoint:        {ckpt_path}")
    print(f"  Checkpoint epoch:  {ckpt.get('epoch', 'N/A')}")
    print(f"  Device:            {device}")
    print(f"  Test images:       {num_to_process}")
    print(f"  Saved outputs:     {saved_count}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'  ':-<62}")
    print(f"  Total time:        {total_time:.2f} s")
    print(f"  Avg per image:     {avg_time * 1000:.2f} ms")
    print(f"{'  ':-<62}")
    print(f"  Output shape:      (256, 256)")
    print(f"  Output dtype:      float32")
    print(f"  Output range:      [0, 1]")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluate.py - Standalone KLA Submission Evaluation Script

Loads the trained KLARestorationModel, runs inference on all .npy files
in the input directory, and saves restored outputs to the output directory.

Usage:
    python evaluate.py <input_directory> <output_directory>

    python evaluate.py <input_directory> <output_directory> --checkpoint weights/best_model.pth

Example:
    python evaluate.py test_images/ restored_outputs/

Checkpoint resolution (in order):
    1. Explicit --checkpoint argument
    2. weights/best_model.pth (relative to this script's directory)

Requirements:
    - PyTorch
    - NumPy
    - The src/model.py module must be importable (run from the repo root)
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve project root so src.model is importable even if cwd differs
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.model import KLARestorationModel  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = _SCRIPT_DIR / "weights" / "best_model.pth"
EXPECTED_INPUT_SHAPE = (128, 128)
EXPECTED_OUTPUT_SHAPE = (256, 256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(user_path: str | None) -> Path:
    """Resolve the checkpoint path.

    Priority:
        1. User-supplied --checkpoint path
        2. weights/best_model.pth relative to this script
    """
    if user_path is not None:
        p = Path(user_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT

    raise FileNotFoundError(
        f"No checkpoint found. Looked at:\n"
        f"  {DEFAULT_CHECKPOINT}\n"
        f"Provide one with --checkpoint <path>"
    )


def _validate_output(arr: np.ndarray, name: str) -> None:
    """Validate a single restored output array."""
    if arr.shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"Output '{name}' shape {arr.shape}, expected {EXPECTED_OUTPUT_SHAPE}"
        )
    if arr.dtype != np.float32:
        raise ValueError(
            f"Output '{name}' dtype {arr.dtype}, expected float32"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Output '{name}' contains NaN or Inf")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"Output '{name}' values outside [0,1]: "
            f"min={arr.min():.6f}, max={arr.max():.6f}"
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KLA AI-Based Image Restoration - Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python evaluate.py test_images/ restored_outputs/\n"
            "  python evaluate.py test_images/ restored_outputs/ "
            "--checkpoint weights/best_model.pth\n"
        ),
    )
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing input .npy test images.",
    )
    parser.add_argument(
        "output_dir",
        help="Path to the directory where restored .npy outputs will be saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to the model checkpoint. "
            "Default: weights/best_model.pth (relative to the repo root)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run evaluation. Returns 0 on success, 1 on error."""
    args = parse_args()

    print("=" * 64)
    print("KLA AI-Based Image Restoration - Evaluation")
    print("=" * 64)

    # ---- Validate input directory ----------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"[ERROR] No .npy files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"[eval] Input directory:  {input_dir}")
    print(f"[eval] Input files:      {len(npy_files)}")

    # ---- Resolve checkpoint ----------------------------------------------
    try:
        ckpt_path = _resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"[eval] Checkpoint:       {ckpt_path}")

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device:           {device}")

    # ---- Load model ------------------------------------------------------
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = KLARestorationModel()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        return 1

    print(f"[eval] Model loaded successfully")
    if "epoch" in ckpt:
        print(f"[eval]   Checkpoint epoch: {ckpt['epoch']}")

    # ---- Output directory ------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Output directory: {output_dir}")

    # ---- Inference -------------------------------------------------------
    print(f"\n[eval] Processing {len(npy_files)} images ...")

    success_count = 0
    error_count = 0
    all_times: list[float] = []

    with torch.no_grad():
        for idx, npy_path in enumerate(npy_files):
            stem = npy_path.stem

            try:
                # Load input
                input_arr = np.load(npy_path)

                if input_arr.shape != EXPECTED_INPUT_SHAPE:
                    raise ValueError(
                        f"Input '{stem}' shape {input_arr.shape}, "
                        f"expected {EXPECTED_INPUT_SHAPE}"
                    )

                # Prepare tensor: (128,128) -> [1, 1, 128, 128]
                input_tensor = (
                    torch.from_numpy(input_arr.astype(np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                # Inference
                t_start = time.time()
                prediction = model(input_tensor)  # [1, 1, 256, 256]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_end = time.time()

                all_times.append(t_end - t_start)

                # Post-process: CPU -> float32 -> clamp [0,1] -> (256,256)
                restored = prediction.detach().cpu().float()
                restored = torch.clamp(restored, 0.0, 1.0)
                restored_np = restored.squeeze(0).squeeze(0).numpy()

                # Validate
                _validate_output(restored_np, stem)

                # Save
                save_path = output_dir / f"{stem}.npy"
                np.save(save_path, restored_np)
                success_count += 1

            except Exception as e:
                print(f"  [ERROR] {stem}: {e}", file=sys.stderr)
                error_count += 1
                continue

            # Progress
            if (idx + 1) % 50 == 0 or (idx + 1) == len(npy_files):
                print(f"  [{idx + 1}/{len(npy_files)}] {stem}.npy saved")

    # ---- Summary ---------------------------------------------------------
    total_time = sum(all_times) if all_times else 0.0
    avg_time = total_time / max(success_count, 1)

    print(f"\n{'=' * 64}")
    print(f"Evaluation Summary")
    print(f"{'=' * 64}")
    print(f"  Input files:       {len(npy_files)}")
    print(f"  Outputs saved:     {success_count}")
    print(f"  Errors:            {error_count}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'  ':-<62}")
    print(f"  Total time:        {total_time:.2f} s")
    print(f"  Avg per image:     {avg_time * 1000:.2f} ms")
    print(f"{'  ':-<62}")
    print(f"  Output shape:      {EXPECTED_OUTPUT_SHAPE}")
    print(f"  Output dtype:      float32")
    print(f"  Output range:      [0, 1]")
    print(f"{'=' * 64}")

    if error_count > 0:
        print(f"\n[WARNING] {error_count} image(s) failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

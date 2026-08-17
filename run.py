#!/usr/bin/env python3
"""
run.py - Standalone KLA Submission Inference Script

Loads the final trained KLARestorationModel, reads every .npy file from
the input directory, restores each image, and writes one .npy output
with the same filename to the output directory.

Usage:
    python run.py <input_directory> <output_directory>

Example:
    python run.py test_images restored_outputs

Expected input:
    Shape: (128, 128)
    Dtype: float32 (or safely convertible to float32)

Generated output:
    Shape: (256, 256)
    Dtype: float32
    Range: [0, 1]

The script:
    - Automatically uses weights/best_model.pth
    - Uses CUDA if available, otherwise CPU
    - Requires no internet during inference
    - Requires no API keys
    - Requires no user interaction
    - Creates the output directory automatically
    - Preserves input filenames
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.model import KLARestorationModel  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = SCRIPT_DIR / "weights" / "best_model.pth"

EXPECTED_INPUT_SHAPE = (128, 128)
EXPECTED_OUTPUT_SHAPE = (256, 256)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(arr: np.ndarray, filename: str) -> None:
    """Validate one restored output."""

    if arr.shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"{filename}: output shape {arr.shape}, "
            f"expected {EXPECTED_OUTPUT_SHAPE}"
        )

    if arr.dtype != np.float32:
        raise ValueError(
            f"{filename}: output dtype {arr.dtype}, expected float32"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"{filename}: output contains NaN or Inf"
        )

    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"{filename}: output values outside [0,1] "
            f"(min={arr.min():.6f}, max={arr.max():.6f})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python run.py <input_directory> <output_directory>\n\n"
            "Example:\n"
            "  python run.py test_images restored_outputs"
        )
        return 1

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    print("=" * 64)
    print("KLA AI-Based Image Restoration - Final Inference")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # Validate input directory
    # -----------------------------------------------------------------------

    if not input_dir.is_dir():
        print(
            f"[ERROR] Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 1

    npy_files = sorted(input_dir.glob("*.npy"))

    if not npy_files:
        print(
            f"[ERROR] No .npy files found in: {input_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"[run] Input directory : {input_dir}")
    print(f"[run] Input files     : {len(npy_files)}")

    # -----------------------------------------------------------------------
    # Validate checkpoint
    # -----------------------------------------------------------------------

    if not CHECKPOINT_PATH.is_file():
        print(
            f"[ERROR] Model checkpoint not found:\n"
            f"        {CHECKPOINT_PATH}",
            file=sys.stderr,
        )
        return 1

    print(f"[run] Checkpoint      : {CHECKPOINT_PATH}")

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[run] Device           : {device}")

    if device.type == "cuda":
        print(
            f"[run] GPU              : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model = KLARestorationModel()

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device)
        model.eval()

    except Exception as exc:
        print(
            f"[ERROR] Failed to load model: {exc}",
            file=sys.stderr,
        )
        return 1

    print("[run] Model loaded successfully")

    if "epoch" in checkpoint:
        print(
            f"[run] Checkpoint epoch: {checkpoint['epoch']}"
        )

    if "val_loss" in checkpoint:
        print(
            f"[run] Validation loss : {checkpoint['val_loss']:.6f}"
        )

    # -----------------------------------------------------------------------
    # Create output directory
    # -----------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[run] Output directory: {output_dir}")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    success_count = 0
    error_count = 0
    total_inference_time = 0.0

    print(
        f"\n[run] Processing {len(npy_files)} input images ..."
    )

    with torch.no_grad():

        for index, input_path in enumerate(npy_files, start=1):

            filename = input_path.name
            stem = input_path.stem

            try:

                # -----------------------------------------------------------
                # Load input
                # -----------------------------------------------------------

                input_array = np.load(input_path)

                if input_array.shape != EXPECTED_INPUT_SHAPE:
                    raise ValueError(
                        f"input shape {input_array.shape}, "
                        f"expected {EXPECTED_INPUT_SHAPE}"
                    )

                input_array = input_array.astype(
                    np.float32,
                    copy=False,
                )

                # -----------------------------------------------------------
                # Prepare tensor
                # (128,128) -> (1,1,128,128)
                # -----------------------------------------------------------

                input_tensor = (
                    torch.from_numpy(input_array)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                # -----------------------------------------------------------
                # Model inference
                # -----------------------------------------------------------

                start_time = time.perf_counter()

                prediction = model(input_tensor)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                total_inference_time += elapsed

                # -----------------------------------------------------------
                # Post-processing
                # -----------------------------------------------------------

                restored = (
                    prediction
                    .detach()
                    .cpu()
                    .float()
                )

                restored = torch.clamp(
                    restored,
                    0.0,
                    1.0,
                )

                restored_array = (
                    restored
                    .squeeze(0)
                    .squeeze(0)
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                # -----------------------------------------------------------
                # Validate output
                # -----------------------------------------------------------

                validate_output(
                    restored_array,
                    filename,
                )

                # -----------------------------------------------------------
                # Save with SAME filename
                # -----------------------------------------------------------

                output_path = (
                    output_dir / filename
                )

                np.save(
                    output_path,
                    restored_array,
                )

                success_count += 1

                # -----------------------------------------------------------
                # Progress
                # -----------------------------------------------------------

                if (
                    index % 50 == 0
                    or index == len(npy_files)
                ):
                    print(
                        f"  [{index}/{len(npy_files)}] "
                        f"{filename} saved "
                        f"({elapsed * 1000:.2f} ms)"
                    )

            except Exception as exc:

                error_count += 1

                print(
                    f"  [ERROR] {filename}: {exc}",
                    file=sys.stderr,
                )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    average_time = (
        total_inference_time / success_count
        if success_count > 0
        else 0.0
    )

    print("\n" + "=" * 64)
    print("KLA Inference Summary")
    print("=" * 64)

    print(
        f"  Input files:       {len(npy_files)}"
    )

    print(
        f"  Outputs saved:     {success_count}"
    )

    print(
        f"  Errors:            {error_count}"
    )

    print(
        f"  Output directory:  {output_dir}"
    )

    print("-" * 64)

    print(
        f"  Output shape:      {EXPECTED_OUTPUT_SHAPE}"
    )

    print(
        f"  Output dtype:      float32"
    )

    print(
        f"  Output range:      [0, 1]"
    )

    print(
        f"  Total inference:   {total_inference_time:.2f} s"
    )

    print(
        f"  Avg per image:     "
        f"{average_time * 1000:.2f} ms"
    )

    print("=" * 64)

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------

    if error_count > 0:
        print(
            f"\n[FAILED] {error_count} image(s) could not be processed.",
            file=sys.stderr,
        )
        return 1

    if success_count != len(npy_files):
        print(
            "\n[FAILED] Not every input produced an output.",
            file=sys.stderr,
        )
        return 1

    print(
        "\n[OK] All input images were restored successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
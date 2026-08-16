"""
src/metrics.py - Evaluation Metrics for KLA Image Restoration

Provides reusable metric functions for comparing restored images against
ground-truth targets:

    mse(prediction, target)   - Mean Squared Error (lower is better)
    psnr(prediction, target)  - Peak Signal-to-Noise Ratio (higher is better)
    ssim(prediction, target)  - Structural Similarity Index (higher is better)

Expected inputs:
    prediction: [H, W] or [1, H, W] or [B, 1, H, W], float32
    target:     same shape as prediction, float32, values in [0, 1]

Prediction clipping policy:
    Model outputs may be outside [0, 1].  Before computing PSNR and SSIM the
    prediction is **clamped to [0, 1]** so that the metrics reflect the quality
    of the final deliverable image.  MSE is computed on the raw values to aid
    training diagnostics.  This choice is documented in each function.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert a torch Tensor or NumPy array to a float32 NumPy array.

    Squeezes leading batch/channel dimensions so the result is 2-D [H, W].
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float32)

    # Squeeze to 2-D: [B, C, H, W] -> [H, W] or [C, H, W] -> [H, W]
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(
            f"Expected a 2-D image after squeezing, got shape {x.shape}. "
            f"Provide a single-image tensor of shape [H,W], [1,H,W], or [1,1,H,W]."
        )
    return x


def _validate_shapes(
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    """Raise if prediction and target shapes do not match."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def mse(
    prediction: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
) -> float:
    """Compute Mean Squared Error between prediction and target.

    MSE is computed on **raw** (unclamped) values.  This is useful during
    training to see how far the model output deviates from the target,
    including any out-of-range values.

    Parameters
    ----------
    prediction : array-like
        Restored image, shape [H,W] / [1,H,W] / [1,1,H,W].
    target : array-like
        Ground-truth image, same shape, values in [0, 1].

    Returns
    -------
    float
        Mean squared error (lower is better).
    """
    pred_np = _to_numpy(prediction)
    tgt_np = _to_numpy(target)
    _validate_shapes(pred_np, tgt_np)

    return float(np.mean((pred_np - tgt_np) ** 2))


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    prediction: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Before computing PSNR the prediction is **clamped to [0, 1]** so the
    metric reflects the quality of the final output image.

    PSNR = 10 * log10(data_range^2 / MSE)

    If MSE is zero (identical images), returns ``float('inf')``.

    Parameters
    ----------
    prediction : array-like
        Restored image, shape [H,W] / [1,H,W] / [1,1,H,W].
    target : array-like
        Ground-truth image, same shape, values in [0, 1].
    data_range : float
        Dynamic range of the target image (default: 1.0 for normalised GT).

    Returns
    -------
    float
        PSNR in dB (higher is better).
    """
    pred_np = _to_numpy(prediction)
    tgt_np = _to_numpy(target)
    _validate_shapes(pred_np, tgt_np)

    # Clamp prediction to valid output range before quality measurement
    pred_np = np.clip(pred_np, 0.0, 1.0)

    mse_val = float(np.mean((pred_np - tgt_np) ** 2))

    if mse_val == 0.0:
        return float("inf")

    return float(10.0 * math.log10((data_range ** 2) / mse_val))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def ssim(
    prediction: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (SSIM).

    Before computing SSIM the prediction is **clamped to [0, 1]** so the
    metric reflects the quality of the final output image.

    Uses ``skimage.metrics.structural_similarity`` when available.

    Parameters
    ----------
    prediction : array-like
        Restored image, shape [H,W] / [1,H,W] / [1,1,H,W].
    target : array-like
        Ground-truth image, same shape, values in [0, 1].
    data_range : float
        Dynamic range of the target image (default: 1.0 for normalised GT).

    Returns
    -------
    float
        SSIM value in [-1, 1] (higher is better; 1.0 = identical).
    """
    pred_np = _to_numpy(prediction)
    tgt_np = _to_numpy(target)
    _validate_shapes(pred_np, tgt_np)

    # Clamp prediction to valid output range before quality measurement
    pred_np = np.clip(pred_np, 0.0, 1.0)

    if not _HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for SSIM. "
            "Install it with: pip install scikit-image"
        )

    return float(_skimage_ssim(pred_np, tgt_np, data_range=data_range))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Metrics -- self-test")
    print("=" * 60)

    # Create a deterministic test image in [0, 1]
    np.random.seed(0)
    img = np.random.rand(256, 256).astype(np.float32)

    # --- Identical images ---
    print("\n--- Identical images ---")
    mse_val = mse(img, img)
    psnr_val = psnr(img, img)
    print(f"MSE:  {mse_val}")
    print(f"PSNR: {psnr_val}")
    assert mse_val == 0.0, f"MSE of identical images should be 0, got {mse_val}"
    assert psnr_val == float("inf"), f"PSNR of identical images should be inf, got {psnr_val}"

    if _HAS_SKIMAGE:
        ssim_val = ssim(img, img)
        print(f"SSIM: {ssim_val}")
        assert abs(ssim_val - 1.0) < 1e-6, f"SSIM of identical images should be ~1.0, got {ssim_val}"
    else:
        print("SSIM: skipped (scikit-image not installed)")

    # --- Different images ---
    print("\n--- Different images ---")
    noisy = np.clip(img + np.random.randn(256, 256).astype(np.float32) * 0.1, 0, 1)
    mse_diff = mse(noisy, img)
    psnr_diff = psnr(noisy, img)
    print(f"MSE:  {mse_diff:.6f}")
    print(f"PSNR: {psnr_diff:.2f} dB")
    assert mse_diff > 0, "MSE of different images should be > 0"
    assert 0 < psnr_diff < float("inf"), "PSNR should be finite and positive"

    if _HAS_SKIMAGE:
        ssim_diff = ssim(noisy, img)
        print(f"SSIM: {ssim_diff:.6f}")
        assert 0 < ssim_diff < 1.0, "SSIM of noisy vs clean should be between 0 and 1"

    # --- Torch tensor support ---
    if _HAS_TORCH:
        print("\n--- Torch tensor input ---")
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,256,256]
        tgt_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        mse_t = mse(img_t, tgt_t)
        psnr_t = psnr(img_t, tgt_t)
        print(f"MSE (tensor):  {mse_t}")
        print(f"PSNR (tensor): {psnr_t}")
        assert mse_t == 0.0
        assert psnr_t == float("inf")

    # --- Out-of-range prediction ---
    print("\n--- Out-of-range prediction (clamping test) ---")
    oor_pred = img.copy()
    oor_pred[0, 0] = 1.5   # above 1
    oor_pred[1, 1] = -0.1  # below 0
    psnr_oor = psnr(oor_pred, img)
    print(f"PSNR (out-of-range pred, clamped): {psnr_oor:.2f} dB")
    assert 0 < psnr_oor < float("inf"), "PSNR should be finite after clamping"

    print("\n[OK] All metric self-tests passed")
    print("=" * 60)

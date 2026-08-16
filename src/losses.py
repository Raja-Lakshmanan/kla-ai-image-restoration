"""
src/losses.py - Training Loss Functions for KLA Image Restoration

Baseline reconstruction loss using L1 (Mean Absolute Error).

L1 loss is chosen as the baseline because:
- It directly penalises pixel-level differences between prediction and GT.
- It is more robust to outliers than L2/MSE (less blurring in practice).
- It is the standard starting point for image restoration tasks.

Expected tensor shapes:
    prediction: [B, 1, 256, 256]  (model output, float32)
    target:     [B, 1, 256, 256]  (GT clean image, float32)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationL1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss for image restoration.

    Computes ``mean(|prediction - target|)`` and returns a scalar tensor
    suitable for ``loss.backward()``.

    Parameters
    ----------
    reduction : str
        Reduction mode passed to ``nn.L1Loss`` (default: ``"mean"``).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L1 loss between prediction and target.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape [B, 1, 256, 256], float32.
        target : torch.Tensor
            Ground-truth image of shape [B, 1, 256, 256], float32.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return self.l1(prediction, target)


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian window."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create a 2D Gaussian window for SSIM."""
    _1d_window = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def differentiable_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    val_range: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable SSIM for a batch of images.
    
    Parameters
    ----------
    prediction : torch.Tensor
        Model output of shape [B, C, H, W], float32.
    target : torch.Tensor
        Ground-truth image of shape [B, C, H, W], float32.
    window_size : int
        Size of the Gaussian window.
    val_range : float
        Dynamic range of the images (default 1.0).
        
    Returns
    -------
    torch.Tensor
        Scalar SSIM value for the batch.
    """
    channel = prediction.size(1)
    window = _create_window(window_size, channel).to(prediction.device)

    # Compute local means
    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class CombinedL1SSIMLoss(nn.Module):
    """Combined L1 and SSIM loss for structural/detail preservation.
    
    Loss = alpha * L1 + beta * (1 - SSIM)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Combined L1 + SSIM loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape [B, 1, 256, 256], float32.
        target : torch.Tensor
            Ground-truth image of shape [B, 1, 256, 256], float32.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        l1_loss = self.l1(prediction, target)
        prediction_ssim = torch.clamp(prediction, 0.0, 1.0)
        ssim_val = differentiable_ssim(prediction_ssim, target)
        ssim_loss = 1.0 - ssim_val
        return self.alpha * l1_loss + self.beta * ssim_loss


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RestorationL1Loss -- self-test")
    print("=" * 60)

    # Create random tensors matching verified KLA shapes
    prediction = torch.randn(2, 1, 256, 256, requires_grad=True)
    target = torch.rand(2, 1, 256, 256)  # GT is in [0,1]

    print(f"Prediction shape: {prediction.shape}")
    print(f"Target shape:     {target.shape}")

    # Compute loss
    criterion = RestorationL1Loss()
    loss = criterion(prediction, target)

    print(f"Loss value:       {loss.item():.6f}")
    print(f"Loss shape:       {loss.shape}")
    print(f"Loss is scalar:   {loss.dim() == 0}")

    # Verify backward pass
    loss.backward()
    grad_exists = prediction.grad is not None
    grad_shape = prediction.grad.shape if grad_exists else None

    print(f"Gradient exists:  {grad_exists}")
    print(f"Gradient shape:   {grad_shape}")

    # Assertions
    assert loss.dim() == 0, f"Loss must be scalar, got dim={loss.dim()}"
    assert grad_exists, "Gradient must exist after backward()"
    assert grad_shape == prediction.shape, "Gradient shape must match prediction"

    print(f"\n[OK] RestorationL1Loss: Loss is scalar: {loss.shape}")
    print(f"[OK] RestorationL1Loss: backward() completed successfully")
    
    print("\n--- Testing CombinedL1SSIMLoss ---")
    prediction2 = torch.randn(2, 1, 256, 256, requires_grad=True)
    
    criterion2 = CombinedL1SSIMLoss(alpha=1.0, beta=0.1)
    loss2 = criterion2(prediction2, target)
    
    print(f"Combined Loss value: {loss2.item():.6f}")
    assert loss2.dim() == 0, f"Combined loss must be scalar, got dim={loss2.dim()}"
    
    loss2.backward()
    grad2_exists = prediction2.grad is not None
    assert grad2_exists, "Gradient must exist after Combined loss backward()"
    assert prediction2.grad.shape == prediction2.shape, "Gradient shape must match prediction"
    
    print(f"[OK] CombinedL1SSIMLoss: Loss is scalar: {loss2.shape}")
    print(f"[OK] CombinedL1SSIMLoss: backward() completed successfully")

    print("=" * 60)

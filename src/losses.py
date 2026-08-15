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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RestorationL1Loss -- self-test")
    print("=" * 60)

    # Create random tensors matching verified KLA shapes
    prediction = torch.randn(2, 1, 256, 256, requires_grad=True)
    target = torch.randn(2, 1, 256, 256)

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

    print(f"\n[OK] Loss is scalar: {loss.shape}")
    print(f"[OK] backward() completed successfully")
    print("=" * 60)

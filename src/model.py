"""
src/model.py - Lightweight Residual CNN for KLA Image Restoration

Architecture: Residual CNN with 2x PixelShuffle upsampling.
    Input:  [B, 1, 128, 128]  (grayscale degraded low-resolution)
    Output: [B, 1, 256, 256]  (grayscale restored high-resolution)

Design choices:
- Residual blocks for stable gradient flow and restoration capacity.
- PixelShuffle (sub-pixel convolution) for learnable, artifact-free 2x upsampling.
- Lightweight enough to train on a Colab T4 GPU (~630K parameters).
- No output clamping — raw float tensor output.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Conv-ReLU-Conv residual block with identity skip connection.

    Preserves spatial dimensions and channel count.

    Parameters
    ----------
    channels : int
        Number of input and output feature channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class KLARestorationModel(nn.Module):
    """Lightweight residual CNN with PixelShuffle upsampling.

    Architecture overview::

        Input [B,1,128,128]
            │
            ▼
        ┌──────────────────────┐
        │  Input Conv (1→64)   │  ← extract initial features
        └──────────┬───────────┘
                   │ (skip_global)
            ┌──────▼──────┐
            │  8 Residual │
            │    Blocks   │  ← restore degraded features
            │  (64→64)    │
            └──────┬──────┘
                   │
            ┌──────▼──────────────┐
            │  Fusion Conv (64→64)│  ← combine with skip_global
            └──────┬──────────────┘
                   │ + skip_global
            ┌──────▼──────────────────┐
            │  Upsample Conv (64→4)   │
            │  PixelShuffle(2)        │  ← 128×128 → 256×256
            └──────┬──────────────────┘
                   │
            ┌──────▼──────────────┐
            │  Output Conv (1→1)  │  ← final refinement
            └──────┬──────────────┘
                   │
        Output [B,1,256,256]

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1 for grayscale).
    num_features : int
        Number of feature channels in the residual body (default: 64).
    num_residual_blocks : int
        Number of residual blocks (default: 8).
    upscale_factor : int
        Spatial upsampling factor (default: 2, i.e. 128→256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 64,
        num_residual_blocks: int = 8,
        upscale_factor: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.num_residual_blocks = num_residual_blocks
        self.upscale_factor = upscale_factor

        # -- Input feature extraction --------------------------------------
        self.input_conv = nn.Conv2d(
            in_channels, num_features, kernel_size=3, padding=1, bias=True,
        )

        # -- Residual body -------------------------------------------------
        self.residual_body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        )

        # -- Feature fusion (after residual body, before global skip add) --
        self.fusion_conv = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1, bias=True,
        )

        # -- 2x PixelShuffle upsampling ------------------------------------
        # PixelShuffle(r) rearranges [B, C*r^2, H, W] → [B, C, H*r, W*r]
        # For 1 output channel with r=2: need 1 * 2^2 = 4 channels before shuffle
        upsample_channels = in_channels * (upscale_factor ** 2)
        self.upsample = nn.Sequential(
            nn.Conv2d(
                num_features, upsample_channels, kernel_size=3, padding=1, bias=True,
            ),
            nn.PixelShuffle(upscale_factor),  # [B, 4, 128, 128] → [B, 1, 256, 256]
        )

        # -- Output refinement ---------------------------------------------
        # Light 3x3 conv to refine the upsampled output
        self.output_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Degraded input of shape [B, 1, 128, 128], float32.
            Values are preserved as-is (may be outside [0, 1]).

        Returns
        -------
        torch.Tensor
            Restored output of shape [B, 1, 256, 256], float32.
            Raw values — no clamping applied.
        """
        # Extract initial features
        shallow_features = self.input_conv(x)           # [B, 64, 128, 128]

        # Residual body
        deep_features = self.residual_body(shallow_features)  # [B, 64, 128, 128]

        # Fusion with global skip connection
        fused = self.fusion_conv(deep_features) + shallow_features  # [B, 64, 128, 128]

        # 2x upsampling via PixelShuffle
        upsampled = self.upsample(fused)                # [B, 1, 256, 256]

        # Output refinement
        output = self.output_conv(upsampled)            # [B, 1, 256, 256]

        return output


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: KLARestorationModel) -> str:
    """Return a human-readable summary string of the model configuration."""
    n_params = count_parameters(model)
    return (
        f"KLARestorationModel\n"
        f"  Input channels:      {model.in_channels}\n"
        f"  Feature channels:    {model.num_features}\n"
        f"  Residual blocks:     {model.num_residual_blocks}\n"
        f"  Upscale factor:      {model.upscale_factor}\n"
        f"  Trainable params:    {n_params:,}\n"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("KLARestorationModel — self-test")
    print("=" * 60)

    model = KLARestorationModel()

    # Print model summary
    print(get_model_summary(model))

    # Create a dummy batch matching verified KLA dataset properties
    dummy_input = torch.randn(2, 1, 128, 128)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Input dtype:  {dummy_input.dtype}")

    # Forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Output shape: {dummy_output.shape}")
    print(f"Output dtype: {dummy_output.dtype}")

    # Verify shapes
    expected_output_shape = torch.Size([2, 1, 256, 256])
    assert dummy_output.shape == expected_output_shape, (
        f"SHAPE MISMATCH: got {dummy_output.shape}, "
        f"expected {expected_output_shape}"
    )

    print(f"\n[OK] Output shape verified: {dummy_output.shape}")
    print(f"[OK] Parameter count: {count_parameters(model):,}")
    print("=" * 60)

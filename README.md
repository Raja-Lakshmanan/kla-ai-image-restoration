# KLA Hackathon 2026 — AI-Based Restoration of Degraded Images

AI-powered restoration of noisy, low-resolution semiconductor inspection images using a lightweight residual CNN with 2× PixelShuffle super-resolution.

## Problem Statement

The goal is to transform degraded NoisyLR semiconductor inspection images into high-quality, analysis-ready images by:

- Suppressing noise
- Recovering fine spatial details
- Enhancing image quality
- Reconstructing images from 128×128 to 256×256

## Final Model

Architecture:

- Grayscale input/output
- Input: `128×128`
- Output: `256×256`
- 8 residual blocks
- 64 feature channels
- 2× PixelShuffle upsampling
- Trainable parameters: `630,734`

Final checkpoint:

```text
weights/best_model.pth
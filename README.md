# KLA Hackathon 2026 — AI-Based Restoration of Degraded Images

AI-powered restoration of noisy, low-resolution semiconductor inspection images using a lightweight residual CNN with 2× PixelShuffle super-resolution.

---

## 1. Problem Statement

The goal of this project is to restore degraded semiconductor inspection images by:

- Suppressing noise
- Recovering important image structures
- Improving visual quality
- Reconstructing images from `128×128` to `256×256`

The model takes a degraded low-resolution image as input and generates a restored high-resolution image.

---

## 2. Solution Overview

The implemented pipeline is:

```text
NoisyLR Image
128×128
    │
    ▼
Input Convolution
    │
    ▼
8 Residual Blocks
    │
    ▼
Feature Fusion
    │
    ▼
2× PixelShuffle
    │
    ▼
Output Refinement
    │
    ▼
Restored Image
256×256


## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Raja-Lakshmanan/kla-ai-image-restoration.git
cd kla-ai-image-restoration

#Install dependencies

python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

#Verify pytorch and cuda

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"


#Run the Standalone Evaluation Script
python evaluate.py <input_directory> <output_directory>
example:python evaluate.py test_images restored_outputs
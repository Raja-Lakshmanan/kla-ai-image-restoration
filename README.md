# KLA Hackathon 2026 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

A lightweight residual CNN that restores noisy, low-resolution semiconductor inspection images (128×128 grayscale) to clean, high-resolution outputs (256×256) using learned 2× PixelShuffle super-resolution. Trained on 3,200 paired samples and evaluated on 320 held-out validation images, the model achieves **28.60 dB PSNR** and **0.760 SSIM** on the validation set.

| Status | |
|---|---|
| Final model checkpoint | `weights/best_model.pth` |
| Official inference entry point | `run.py` |
| Restored test outputs | 400 files in `restored_test_outputs/` |
| Validation metrics | `results/validation_metrics.csv` |
| Repository | Public on GitHub |

> **Hackathon Evaluator — run this:**
> ```bash
> python run.py <input_directory> <output_directory>
> ```

---

**Team Techtrons** — PSG College of Technology

| Role | Name |
|---|---|
| Team Leader | Raja Lakshmanan A |
| Member | Nithishkumar M |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Objective](#2-project-objective)
3. [Solution Overview](#3-solution-overview)
4. [Model Architecture](#4-model-architecture)
5. [Model Flow](#5-model-flow)
6. [Dataset](#6-dataset)
7. [Data Preprocessing](#7-data-preprocessing)
8. [Training Pipeline](#8-training-pipeline)
9. [Loss Function](#9-loss-function)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Final Model](#11-final-model)
12. [Verified Validation Results](#12-verified-validation-results)
13. [Qualitative Results](#13-qualitative-results)
14. [Final Test Inference (run.py)](#14-final-test-inference-runpy)
15. [Input Format for run.py](#15-input-format-for-runpy)
16. [Output Format for run.py](#16-output-format-for-runpy)
17. [Model Weights](#17-model-weights)
18. [Restored Test Outputs](#18-restored-test-outputs)
19. [Quick Start](#19-quick-start)
20. [Installation](#20-installation)
21. [Training from Scratch](#21-training-from-scratch)
22. [Validation Evaluation](#22-validation-evaluation)
23. [Test Inference from ZIP](#23-test-inference-from-zip)
24. [Reproducibility](#24-reproducibility)
25. [Technology Stack](#25-technology-stack)
26. [Project Structure](#26-project-structure)
27. [Limitations](#27-limitations)
28. [Results Summary](#28-results-summary)
29. [FAQ](#29-faq)
30. [References](#30-references)

---

## 1. Problem Statement

Semiconductor manufacturing relies on high-resolution inspection imagery to detect defects at the nanometer scale. In practice, inspection images can be degraded by sensor noise, limited photon budgets, and low spatial resolution — reducing the ability of both human inspectors and automated systems to identify critical features.

This project addresses the challenge of **restoring degraded semiconductor inspection images**:

- **Suppressing noise** that obscures fine structural details
- **Recovering spatial resolution** (2× upsampling from 128×128 to 256×256)
- **Preserving important image structures** such as edges, patterns, and defect signatures
- **Producing clean outputs** suitable for downstream inspection and analysis

Restored images can improve the reliability of subsequent defect detection, classification, and measurement pipelines.

---

## 2. Project Objective

Build a deep learning model that learns the mapping from degraded inputs to high-quality target images using paired training samples.

| | Specification |
|---|---|
| **Input** | 128×128 single-channel grayscale degraded image |
| **Output** | 256×256 single-channel grayscale restored image |
| **Task** | Joint denoising + 2× super-resolution |
| **Learning** | Supervised, from paired (degraded, ground-truth) samples |

---

## 3. Solution Overview

The restoration pipeline processes each degraded image through the following stages:

```text
128×128 Degraded Image
        │
        ▼
   Input Convolution         ← extract 64 feature maps from raw input
        │
        ▼
   8 Residual Blocks         ← progressively restore degraded features
        │
        ▼
   Feature Fusion + Skip     ← combine deep and shallow features
        │
        ▼
   Upsampling Convolution    ← prepare channels for PixelShuffle
        │
        ▼
   PixelShuffle ×2           ← rearrange features into 2× spatial resolution
        │
        ▼
   Output Refinement         ← light 3×3 conv for final cleanup
        │
        ▼
256×256 Restored Image
```

**How it works in plain language:**

1. **Input Convolution** — A single convolutional layer converts the 1-channel input into 64 feature maps, capturing low-level patterns like edges and textures.
2. **Residual Blocks** — Eight stacked residual blocks (each containing two 3×3 convolutions with a ReLU activation) progressively learn to remove noise and recover details. The identity skip connection inside each block ensures stable gradient flow during training.
3. **Feature Fusion** — A fusion convolution combines the deep features from the residual body with the original shallow features via a global skip connection. This prevents the loss of low-level information during deep processing.
4. **PixelShuffle ×2** — A convolution maps 64 channels to 4 channels, and then `PixelShuffle(2)` rearranges these 4 channels at 128×128 into 1 channel at 256×256. This is a learned upsampling method that avoids the checkerboard artifacts common with transposed convolutions.
5. **Output Refinement** — A final 3×3 convolution refines the upsampled image to produce the clean output.

---

## 4. Model Architecture

The model is implemented in `src/model.py` as `KLARestorationModel`.

| Property | Value |
|---|---|
| Input channels | 1 (grayscale) |
| Output channels | 1 (grayscale) |
| Feature channels | 64 |
| Residual blocks | 8 |
| Upscale factor | 2 (PixelShuffle) |
| **Trainable parameters** | **630,734** |
| Input tensor shape | `[B, 1, 128, 128]` |
| Output tensor shape | `[B, 1, 256, 256]` |

### Residual Block Structure

Each residual block has the following structure:

```text
Input (64 channels)
  │
  ├──→ Conv2d(64, 64, 3×3, padding=1) → ReLU → Conv2d(64, 64, 3×3, padding=1)
  │                                                          │
  └────────────────────── + ─────────────────────────────────┘
                          │
                  Output (64 channels)
```

The identity skip connection (`x + block(x)`) allows gradients to flow directly through the network during training. This means the block only needs to learn the *residual* difference between its input and the desired output, which is easier to optimize than learning the full transformation.

### PixelShuffle Upsampling

`PixelShuffle(r)` rearranges a tensor of shape `[B, C×r², H, W]` into `[B, C, H×r, W×r]`. For this model with `r=2` and 1 output channel, a convolution maps 64 features to 4 channels (`1 × 2² = 4`), and PixelShuffle then rearranges those 4 channels at 128×128 into 1 channel at 256×256.

Unlike simple nearest-neighbor or bilinear upsampling (which apply fixed interpolation), PixelShuffle learns the upsampling mapping from data, potentially producing sharper and more detailed outputs.

### Global Skip Connection

After the residual body, a fusion convolution produces a feature map that is added element-wise to the original shallow features from the input convolution. This global skip connection ensures that the upsampling stage receives both low-level detail (edges, textures) and high-level restoration information.

---

## 5. Model Flow

```text
Input [B, 1, 128, 128]
         │
         ▼
┌────────────────────────┐
│  Input Conv (1 → 64)   │   Conv2d, 3×3, pad=1
└──────────┬─────────────┘
           │ ←── skip_global
     ┌─────▼─────┐
     │ Residual  │
     │ Block ×8  │   Each: Conv-ReLU-Conv + identity skip
     │ (64 → 64) │
     └─────┬─────┘
           │
┌──────────▼─────────────┐
│  Fusion Conv (64 → 64) │   Conv2d, 3×3, pad=1
└──────────┬─────────────┘
           │ + skip_global
┌──────────▼─────────────┐
│  Upsample Conv (64→ 4) │   Conv2d, 3×3, pad=1
│  PixelShuffle(2)       │   [B,4,128,128] → [B,1,256,256]
└──────────┬─────────────┘
           │
┌──────────▼─────────────┐
│  Output Conv (1 → 1)   │   Conv2d, 3×3, pad=1
└──────────┬─────────────┘
           │
Output [B, 1, 256, 256]
```

---

## 6. Dataset

The dataset consists of paired semiconductor inspection images provided for the KLA Hackathon 2026.

| Property | Value |
|---|---|
| Total training pairs | 3,200 |
| Training split | 2,880 (90%) |
| Validation split | 320 (10%) |
| Test images (NoisyLR only) | 400 |
| Input (NoisyLR) resolution | 128×128 |
| Ground-truth (GT) resolution | 256×256 |
| Data type | float32 (`.npy` files) |
| NoisyLR value range | May exceed [0, 1] |
| GT value range | [0, 1] |
| Train/val split seed | 42 |
| Test set ground truth | **Not available** |

**Data loading:** The dataset module (`src/dataset.py`) supports loading from either a ZIP archive (e.g., `train.zip` containing `NoisyLR/` and `GT/` folders) or an extracted directory with the same structure. Data is loaded lazily — one sample at a time — to minimize memory usage. ZIP file handles are opened and closed per-sample to avoid stale-handle issues on Windows during long training runs.

---

## 7. Data Preprocessing

The preprocessing pipeline implemented in the repository is minimal and intentional:

| Step | Detail |
|---|---|
| Shape validation | Each NoisyLR array must be `(128, 128)`, each GT array must be `(256, 256)` |
| dtype validation | Both must be `float32` |
| NaN/Inf check | Arrays containing non-finite values are rejected |
| Channel dimension | Added via `unsqueeze(0)`: `(H, W)` → `(1, H, W)` |
| Normalization | **None** — pixel values are preserved as-is |
| NoisyLR values | Preserved exactly (may be outside [0, 1]) |
| GT values | Preserved in their original [0, 1] range |
| Output clamping | Applied **post-inference** in `run.py`: `torch.clamp(output, 0.0, 1.0)` |

No data augmentation is applied during training or inference.

---

## 8. Training Pipeline

Training is implemented in `train.py` and uses the following configuration:

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Epochs | 50 |
| Random seed | 42 |
| AMP (mixed precision) | Enabled (float16 on CUDA) |
| Checkpointing | Best validation loss model saved |
| Resume support | Supported via `--resume-from` |

**Training loop in plain language:**

```text
For each epoch:
    For each batch of images:
        1. Load degraded images and ground-truth targets
        2. Pass degraded images through the model → prediction
        3. Compute loss between prediction and target
        4. Compute gradients via backpropagation
        5. Update model weights via optimizer step
    After all batches:
        6. Evaluate on validation set
        7. If validation loss improved → save checkpoint
```

The best checkpoint is saved whenever the validation loss decreases, ensuring the final saved model has the lowest validation error observed during training.

---

## 9. Loss Function

### Final Model Loss: L1 (Mean Absolute Error)

The final model (`weights/best_model.pth`) was trained using **L1 loss** (`RestorationL1Loss` in `src/losses.py`):

```
L1 Loss = mean(|prediction - target|)
```

L1 loss directly penalizes pixel-level differences between the model output and the ground-truth image. Compared to L2 (MSE) loss, L1 tends to be more robust to outlier pixels and generally produces less blurry results in image restoration tasks.

### Experimental: Combined L1 + SSIM Loss

The repository also contains `CombinedL1SSIMLoss` in `src/losses.py`, which combines L1 loss with a differentiable SSIM loss:

```
Combined Loss = α × L1 + β × (1 - SSIM)
```

where α = 1.0 and β = 0.1. **This loss was not used to train the final model** and is available for experimental purposes only.

---

## 10. Evaluation Metrics

Three metrics are computed to evaluate restoration quality (implemented in `src/metrics.py`):

| Metric | What It Measures | Better |
|---|---|---|
| **MSE** (Mean Squared Error) | Average squared pixel difference between prediction and target. Computed on **raw** (unclamped) values. | Lower |
| **PSNR** (Peak Signal-to-Noise Ratio) | Ratio between maximum possible signal power and noise power, in decibels. Computed on **clamped** [0, 1] prediction. | Higher |
| **SSIM** (Structural Similarity Index) | Perceptual similarity considering luminance, contrast, and structure. Computed on **clamped** [0, 1] prediction using `skimage.metrics.structural_similarity`. | Higher (max 1.0) |

**Prediction clamping policy:** PSNR and SSIM are computed after clamping the model output to [0, 1], reflecting the quality of the final deliverable image. MSE is computed on raw (unclamped) values to aid training diagnostics.

---

## 11. Final Model

> **This is the model used for all final test inference.**

| Property | Value |
|---|---|
| Checkpoint | `weights/best_model.pth` |
| Best epoch | 48 (out of 50) |
| Training loss (at best epoch) | 0.031369 |
| Validation loss (at best epoch) | 0.028958 |
| Parameters | 630,734 |
| Loss function | L1 (RestorationL1Loss) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| AMP | Enabled |

The checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `train_loss`, `val_loss`, and `config`.

---

## 12. Verified Validation Results

These metrics were computed on the **320 validation images** (10% held-out split, seed=42) using `src/evaluate_validation.py`. Per-image results are saved in `results/validation_metrics.csv`.

| Metric | Value |
|---|---|
| Validation samples | 320 |
| **Mean MSE** | 0.002336 |
| **Mean PSNR** | 28.60 dB |
| **Mean SSIM** | 0.760353 |
| Mean inference time per image | 35.25 ms |

> **Note:** These are **validation-set results**, not hidden test-set benchmark scores. The test set does not include ground-truth images, so PSNR/SSIM cannot be computed on the test outputs. Inference time is hardware-dependent and was measured on a single GPU; results will vary on different hardware.

---

## 13. Qualitative Results

Visual comparison images are available in `results/validation_samples/`. Each image shows a three-panel comparison:

```
Input NoisyLR (128×128)  |  Restored (256×256)  |  Ground Truth (256×256)
```

Five evenly-spaced validation samples are included:

- `comparison_000001.png`
- `comparison_000956.png`
- `comparison_001661.png`
- `comparison_002432.png`
- `comparison_003185.png`

These provide a visual indication of restoration quality across different image types in the validation set.

---

## 14. Final Test Inference (run.py)

### `run.py` is the official KLA submission inference entry point.

```bash
python run.py <input_directory> <output_directory>
```

**Example:**

```bash
python run.py test_images restored_outputs
```

**What `run.py` does:**

1. Reads all `.npy` files from the input directory
2. Loads the checkpoint from `weights/best_model.pth`
3. Automatically selects CUDA if available, otherwise uses CPU
4. Creates the output directory if it does not exist
5. Processes every input image through the model
6. Clamps outputs to [0, 1] and validates shape, dtype, and value range
7. Saves each output as a `.npy` file with the **same filename** as the input
8. Prints a summary with timing information

**What `run.py` does NOT require:**

- No internet connection during inference
- No API keys
- No manual configuration
- No user interaction
- No additional model download (weights are included in the repository)

**One output is produced per input.** If any input fails validation, the script reports the error and returns a non-zero exit code.

---

## 15. Input Format for run.py

| Property | Requirement |
|---|---|
| File format | `.npy` (NumPy array) |
| Shape | `(128, 128)` |
| Channels | Single-channel grayscale |
| dtype | float32 (or safely convertible to float32) |

If an input file has a shape other than `(128, 128)`, `run.py` raises a `ValueError` with a descriptive error message and skips that file.

---

## 16. Output Format for run.py

| Property | Specification |
|---|---|
| File format | `.npy` (NumPy array) |
| Shape | `(256, 256)` |
| dtype | float32 |
| Value range | [0, 1] (clamped) |
| Filename | Same as the corresponding input filename |

**Example:**

| Input | Output |
|---|---|
| `test_images/000298.npy` | `restored_outputs/000298.npy` |

Each output is validated before saving for:
- Correct shape `(256, 256)`
- Correct dtype `float32`
- All values finite (no NaN or Inf)
- All values within [0, 1]

---

## 17. Model Weights

The trained model weights are included in the repository:

```
weights/best_model.pth
```

**No additional download is required.** The inference script `run.py` automatically loads this checkpoint. The file is approximately 7.3 MB.

---

## 18. Restored Test Outputs

Pre-computed restored outputs for all 400 test images are included in:

```
restored_test_outputs/
```

| Property | Value |
|---|---|
| Number of files | 400 |
| Filename range | `000000.npy` to `000399.npy` |
| Shape per file | `(256, 256)` |
| dtype | float32 |
| Value range | [0, 1] |

These outputs were generated using `run.py` with the final checkpoint `weights/best_model.pth`. No ground-truth images are available for the test set, so these outputs have not been evaluated with PSNR/SSIM.

---

## 19. Quick Start

### For Hackathon Evaluators

```bash
# 1. Clone the repository
git clone https://github.com/Raja-Lakshmanan/kla-ai-image-restoration.git
cd kla-ai-image-restoration

# 2. Install dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# 3. Verify PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 4. Run inference
python run.py <input_directory> <output_directory>
```

**Example with test images:**

```bash
python run.py test_images restored_outputs
```

The restored outputs will be saved in the specified output directory. Each output is a `(256, 256)` float32 `.npy` file with values in [0, 1].

---

## 20. Installation

### Requirements

- Python 3.13+
- CUDA-capable GPU (recommended, not required — CPU inference is supported)

### Install Dependencies

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

The `--extra-index-url` flag is required because the PyTorch packages in `requirements.txt` are CUDA 12.8 wheels that are hosted on PyTorch's own package index.

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Key Dependencies

| Package | Version |
|---|---|
| numpy | 2.1.3 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| scikit-image | 0.26.0 |
| matplotlib | 3.11.1 |

---

## 21. Training from Scratch

Training requires access to the KLA training dataset (a ZIP archive or extracted directory containing `NoisyLR/` and `GT/` subfolders).

### Full Training

```bash
python train.py --train-zip path/to/train.zip --epochs 50 --batch-size 4 --lr 1e-4 --weight-decay 1e-4 --amp
```

### Smoke Test (Quick Verification)

```bash
python train.py --train-zip path/to/train.zip --smoke-test
```

Runs 1 epoch on a small subset (32 train, 8 val) to verify the pipeline works end-to-end.

### Resume Training

```bash
python train.py --train-zip path/to/train.zip --epochs 100 --resume-from weights/best_model.pth
```

### All Available Options

| Flag | Default | Description |
|---|---|---|
| `--train-zip` | *(required)* | Path to training ZIP or extracted directory |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--seed` | 42 | Random seed |
| `--loss` | `l1` | Loss function (`l1` or `l1_ssim`) |
| `--checkpoint-path` | `weights/best_model.pth` | Where to save the best checkpoint |
| `--amp` | False | Enable automatic mixed precision |
| `--smoke-test` | False | Run a quick 1-epoch verification |
| `--resume-from` | None | Resume from a checkpoint |
| `--num-workers` | 0 | DataLoader workers |

---

## 22. Validation Evaluation

To reproduce the validation metrics, you need access to the training dataset (to reconstruct the seed=42 validation split):

```bash
python -m src.evaluate_validation \
    --checkpoint weights/best_model.pth \
    --train-zip path/to/train.zip
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to model checkpoint |
| `--train-zip` | *(required)* | Path to training ZIP (for val split) |
| `--seed` | 42 | Random seed (must match training) |
| `--results-dir` | `results` | Output directory for metrics and images |
| `--num-visual` | 5 | Number of visual comparison images |
| `--batch-size` | 1 | Batch size for inference |

### Generated Outputs

| File | Description |
|---|---|
| `results/validation_metrics.csv` | Per-image MSE, PSNR, SSIM, and inference time + aggregate MEAN row |
| `results/validation_samples/` | Side-by-side comparison images (Input \| Restored \| Ground Truth) |

---

## 23. Test Inference from ZIP

The repository includes `src/inference.py` as a development utility for running inference on the official test ZIP archive (`Test_NoisyLR.zip`). This is distinct from `run.py`:

| Script | Purpose |
|---|---|
| **`run.py`** | **Official KLA evaluator entry point** — reads a flat directory of `.npy` files |
| `src/inference.py` | Development utility — reads from a test ZIP archive via the dataset module |

### Usage

```bash
# Full inference (all 400 test images)
python -m src.inference \
    --checkpoint weights/best_model.pth \
    --test-zip path/to/Test_NoisyLR.zip \
    --output-dir results/test_outputs

# Quick test (3 images only)
python -m src.inference \
    --checkpoint weights/best_model.pth \
    --test-zip path/to/Test_NoisyLR.zip \
    --output-dir results/test_outputs \
    --limit 3
```

---

## 24. Reproducibility

| Factor | Implementation |
|---|---|
| Random seed | 42 (Python, NumPy, PyTorch, CUDA) |
| Train/val split | Deterministic with `numpy.random.RandomState(42)` |
| CuDNN | `deterministic=True`, `benchmark=False` |
| Checkpoint | Best validation loss model saved automatically |
| Config saved | Training configuration stored in checkpoint |

To reproduce the exact validation results:

1. Use the same training data and seed=42
2. Train with the same hyperparameters (or use the provided checkpoint)
3. Run `src/evaluate_validation.py` with `--seed 42`

---

## 25. Technology Stack

| Technology | Role |
|---|---|
| Python 3.13 | Programming language |
| PyTorch 2.11.0+cu128 | Deep learning framework |
| NumPy 2.1.3 | Array operations and data I/O |
| scikit-image 0.26.0 | SSIM metric computation |
| Matplotlib 3.11.1 | Validation comparison visualizations |
| CUDA 12.8 | GPU acceleration |
| GitHub | Version control and distribution |

---

## 26. Project Structure

```text
kla-ai-image-restoration/
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── run.py                             # Official inference entry point
├── train.py                           # Training pipeline
│
├── src/
│   ├── __init__.py
│   ├── model.py                       # KLARestorationModel architecture
│   ├── dataset.py                     # Dataset loading (ZIP + directory support)
│   ├── losses.py                      # L1 and experimental L1+SSIM losses
│   ├── metrics.py                     # MSE, PSNR, SSIM evaluation
│   ├── inference.py                   # Test inference utility (from ZIP)
│   ├── evaluate_validation.py         # Validation evaluation pipeline
│   └── utils.py                       # Utility placeholder
│
├── configs/
│   └── config.yaml                    # Configuration placeholder
│
├── weights/
│   └── best_model.pth                 # Final trained checkpoint (epoch 48)
│
├── results/
│   ├── validation_metrics.csv         # Per-image + aggregate validation metrics
│   └── validation_samples/            # Visual comparison images (5 samples)
│
└── restored_test_outputs/             # 400 restored test images (.npy)
```

---

## 27. Limitations

- **Validation ≠ test benchmark.** The reported PSNR (28.60 dB) and SSIM (0.760) are measured on the validation set. No ground truth is available for the test set, so official benchmark performance may differ.
- **Grayscale only.** The model processes single-channel grayscale images. It does not handle RGB or multi-channel inputs.
- **Fixed input size.** The model expects exactly 128×128 input. Other resolutions will raise an error.
- **Hardware-dependent speed.** Inference timing varies by GPU model, CPU, and system load.
- **Training distribution.** Generalization to images outside the KLA training distribution is not guaranteed.
- **No normalization.** Input pixel values are preserved as-is. The model was trained on data with specific value distributions.

---

## 28. Results Summary

| Item | Verified Value |
|---|---|
| Input resolution | 128×128 grayscale |
| Output resolution | 256×256 grayscale |
| Model parameters | 630,734 |
| Validation samples | 320 |
| Best checkpoint epoch | 48 |
| Validation loss (L1) | 0.028958 |
| Mean MSE | 0.002336 |
| Mean PSNR | 28.60 dB |
| Mean SSIM | 0.760353 |
| Final test outputs | 400 |
| Inference entry point | `run.py` |

---

## 29. FAQ

**Q: Do I need to train the model before running inference?**
A: No. The final trained checkpoint (`weights/best_model.pth`) is already included in the repository.

**Q: What command should the evaluator run?**
A: `python run.py <input_directory> <output_directory>`

**Q: What input format is required?**
A: `.npy` files with shape `(128, 128)`, dtype float32.

**Q: What output is generated?**
A: `.npy` files with shape `(256, 256)`, dtype float32, values in [0, 1]. One output per input, same filename.

**Q: Does inference require internet access?**
A: No, assuming all dependencies are already installed.

**Q: Does inference require an API key?**
A: No.

**Q: Which model checkpoint is used?**
A: `weights/best_model.pth` — loaded automatically by `run.py`.

**Q: Are the reported PSNR/SSIM values from the hidden test set?**
A: No. They are computed on the 320-image validation set (10% held-out split, seed=42). The test set has no ground truth.

**Q: Can the model run on CPU?**
A: Yes. `run.py` automatically falls back to CPU if CUDA is not available. Inference will be slower but functionally identical.

**Q: What loss function was used to train the final model?**
A: L1 loss (Mean Absolute Error).

---

## 30. References

1. **Residual Learning for Image Recognition** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. Introduced residual connections for stable deep network training.

2. **Sub-Pixel Convolution (PixelShuffle)** — Shi, W., Caballero, J., Huszár, F., et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.* CVPR 2016. Introduced the PixelShuffle operation for efficient learned upsampling.

3. **Structural Similarity (SSIM)** — Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing, 13(4), 600–612.

---

*KLA Hackathon 2026 — Team Techtrons — PSG College of Technology*
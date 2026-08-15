# KLA Hackathon 2026: AI-Based Restoration of Degraded Images

This repository is for the KLA Hackathon 2026 problem statement: **AI-Based Restoration of Degraded Images**.

## Overview
The goal of this project is to develop an AI-based model to restore degraded images (such as semiconductor inspection images) to high quality.

## Project Structure
- `configs/`: Configuration files for training and evaluation pipeline settings.
- `src/`: Core Python modules (dataset loading, model architecture, loss functions, metrics, utilities).
- `train.py`: Script to train the restoration model from scratch.
- `evaluate.py`: Standalone evaluation script to run model inference on test images and output restored images.
- `weights/`: Directory reserved for trained model weights.
- `restored_test_outputs/`: Directory reserved for saving model-restored test image outputs.
- `results/`: Directory reserved for quantitative and qualitative evaluation results.
- `requirements.txt`: Environment dependencies required to run training and evaluation.

## Setup & Instructions
*(To be updated after model implementation and environment finalization)*

## TODO
- [ ] Finalize PyTorch Dataset and DataLoader pipelines in `src/dataset.py`.
- [ ] Implement AI model architecture in `src/model.py`.
- [ ] Define training loss functions in `src/losses.py`.
- [ ] Define evaluation metrics (e.g., PSNR, SSIM) in `src/metrics.py`.
- [ ] Implement utility helper functions in `src/utils.py`.
- [ ] Configure experiment parameters in `configs/config.yaml`.
- [ ] Build end-to-end reproducible training script in `train.py`.
- [ ] Build standalone evaluation and inference script in `evaluate.py`.
- [ ] Populate `requirements.txt` with exact pinned dependencies.
- [ ] Save trained model weights to `weights/`.
- [ ] Generate and verify restored test outputs in `restored_test_outputs/`.

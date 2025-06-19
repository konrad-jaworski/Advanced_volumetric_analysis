# CNN-ViT vs Attention Unet for Volumetric Data Segmentation

## Overview

This repository presents the implementation, training, and evaluation of two deep learning models for volumetric data segmentation:

- A classic **3D U-Net**
- A custom hybrid model combining **CNN and Vision Transformer (CNN-ViT)**

The entire comparison, analysis, and visualization of results are documented in **`Main_notebook.ipynb`**.

---

## Highlights

- ✅ Successfully trained and evaluated two 3D models capable of processing **variable-length volumetric sequences**.
- ✅ Adapted both architectures to support **sequence depth flexibility**, which is critical for real-world 3D datasets.
- ✅ Developed a custom Transformer-based segmentation pipeline integrated with CNN-based feature extraction.
- ✅ Implemented metric evaluation including **Dice score, IoU, F1-score, Precision, and Recall**.
- ✅ Modular, well-documented PyTorch codebase with separate components for:
  - Feature extraction
  - Transformer attention blocks
  - Positional encoding and upsampling
  - Evaluation pipeline

---

## Dataset

Due to the lack of access to thermographic data, we used a publicly available **CT scan dataset** as a substitute:

**🗂 Source**: [COVID-19 CT Scans — Kaggle](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans)

- The dataset includes volumetric CT scans labeled for COVID-19 infection.
- Each scan was treated as a 3D volume input with variable depth to simulate sequences similar to thermographic recordings.

---

## Repository Structure

```bash
.
├── models/
│   ├── networks/
│   └── layers/
├── train/
│   ├── train_cnn_vit.py
│   └── metrics.py
├── checkpoints/
├── data/
├── Main_notebook.ipynb    ← Full training + evaluation workflow
└── README.md

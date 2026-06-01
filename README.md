# HFSA-Net: Hybrid Frequency-Spatial Attention Network for Low-Light Image Enhancement

**Melahat Bahçeli** — Department of Computer Engineering, Abdullah Gul University, 2026

---

## Overview

HFSA-Net is a deep learning framework for low-light image enhancement that combines:
- **Fourier-based frequency decomposition** (dual-path: low-freq + high-freq)
- **Spatial CNN processing** (dilated encoder-decoder with residual connections)
- **CBAM attention-based feature fusion**
- **Residual reconstruction head**

The model is trained and evaluated on the **LOL dataset** (485 paired images) and outperforms classical (CLAHE) and deep learning (Zero-DCE) baselines on all metrics.

---

## Results

| Method | MSE ↓ | PSNR ↑ (dB) | SSIM ↑ |
|--------|--------|-------------|--------|
| CLAHE | 0.1020 | 10.39 | 0.4288 |
| Zero-DCE | 0.0269 | 16.16 | 0.6857 |
| **HFSA-Net (ours)** | **0.0151** | **18.74** | **0.8903** |

---

## Architecture

```
Low-Light Input
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
Frequency Branch                  Spatial Branch
  2D FFT → [Real | Imag]          Dilated Encoder
  Low-Freq Path (Conv 5×5)        (ResBlocks d=1,2,4)
  High-Freq Path (Conv 3×3)       Decoder
  Concat → [B, 128, H, W]         [B, 128, H, W]
      │                                  │
      └──────────────┬───────────────────┘
                     ▼
           Concat → [B, 256, H, W]
                     ▼
           CBAM Attention Fusion
           (Channel + Spatial Attention)
                     ▼
            1×1 Conv → [B, 128, H, W]
                     ▼
          Reconstruction Head (Conv → Tanh)
                     ▼
           ⊕ Residual Add (+ Input)
                     ▼
           Enhanced Output [B, 3, H, W]
```

---

## Dataset

**LOL (Low-Light) Dataset** — Wei et al., 2018
- 485 paired low-light / normal-light images
- Split: 80% training (388), 20% validation (97)
- Download: [https://daooshee.github.io/BMVC2018website/](https://daooshee.github.io/BMVC2018website/)

Place the dataset in:
```
your_drive/lowlight_project/our485/
├── low/    ← 485 low-light images
└── high/   ← 485 normal-light images
```

---

## Requirements

```bash
pip install torch torchvision pytorch-msssim opencv-python Pillow matplotlib numpy
```

---

## Usage

Open and run the notebook step by step on **Google Colab (T4 GPU)**:

```
hfsa_net_colab.ipynb
```

### Steps in the notebook:
1. GPU check
2. Google Drive mount + dataset verification
3. Install libraries & imports
4. Dataset class (LOLDataset with augmentation)
5. HFSA-Net model architecture
6. Loss function (MSE + SSIM + Perceptual/VGG-16)
7. Optimizer (AdamW) + Metrics
8. Training loop (early stopping)
9. Training curves
10. Visual results
11. Baseline comparison (CLAHE, Zero-DCE)
12. Comparison charts
13. Ablation study
14. Loss weight sensitivity analysis
15. Architecture diagram

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30) |
| Batch size | 8 |
| Input resolution | 256×256 |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Loss weights | 0.4·MSE + 0.4·(1−SSIM) + 0.2·Perceptual |

---

## Ablation Study

| Variant | PSNR (dB) | SSIM |
|---------|-----------|------|
| Frequency Only | 11.36 | 0.4908 |
| No CBAM | 14.76 | 0.7515 |
| Spatial Only | 16.15 | 0.7834 |
| **Full HFSA-Net** | **18.74** | **0.8903** |

---

## Project Structure

```
hfsa-net-low-light-enhancement/
├── HFSA_NET.ipynb     ← Main Colab notebook
└── README.md
```

---

## Acknowledgements

This project was developed as a term project at Abdullah Gul University, Department of Computer Engineering.

AI tools (Claude.ai, ChatGPT) were used for code assistance, language editing, and structure improvement. All generated content was reviewed and verified by the author.

---

## References

- Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex decomposition for low-light enhancement. BMVC.
- Guo, C., et al. (2020). Zero-reference deep curve estimation for low-light image enhancement. CVPR.
- Woo, S., et al. (2018). CBAM: Convolutional block attention module. ECCV.
- Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV.

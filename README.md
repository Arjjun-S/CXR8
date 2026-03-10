# CXR8 - Chest X-Ray Disease Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project for **multi-label chest X-ray disease classification** using the NIH ChestX-ray8 dataset. This repository implements and compares **8 different architectures** including CNNs and Vision Transformers.

![Chest X-Ray Classification](https://img.shields.io/badge/Task-Multi--Label%20Classification-orange)
![Diseases](https://img.shields.io/badge/Classes-15%20Diseases-purple)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Performance Comparison](#performance-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## Overview

This project tackles the challenging problem of **automated thoracic disease detection** from chest X-rays. The NIH ChestX-ray8 dataset presents significant challenges:

- **Multi-label classification**: Images can have multiple disease labels
- **Severe class imbalance**: "No Finding" dominates (~60%), while "Hernia" appears in only 0.02%
- **Subtle visual features**: Many pathologies have overlapping visual characteristics

We implement and compare **8 deep learning architectures** to find the best approach for this medical imaging task.

---

## Dataset

### NIH ChestX-ray8 Dataset

| Attribute | Value |
|-----------|-------|
| **Total Images** | 112,120 |
| **Image Size** | 1024 × 1024 (resized to 224 × 224) |
| **Classes** | 15 (14 diseases + No Finding) |
| **Train/Val Split** | 80% / 20% |

### Disease Classes

| Disease | Samples | Prevalence |
|---------|---------|------------|
| Atelectasis | 11,559 | 10.3% |
| Cardiomegaly | 2,776 | 2.5% |
| Consolidation | 4,667 | 4.2% |
| Edema | 2,303 | 2.1% |
| Effusion | 13,317 | 11.9% |
| Emphysema | 2,516 | 2.2% |
| Fibrosis | 1,686 | 1.5% |
| Hernia | 227 | 0.2% |
| Infiltration | 19,894 | 17.7% |
| Mass | 5,782 | 5.2% |
| Nodule | 6,331 | 5.6% |
| Pleural_Thickening | 3,385 | 3.0% |
| Pneumonia | 1,431 | 1.3% |
| Pneumothorax | 5,302 | 4.7% |
| No Finding | 60,361 | 53.8% |

---

## Models

### 1. DenseNet121 (`Model_DenseNet121/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | DenseNet-121 (pretrained on ImageNet) |
| **Parameters** | ~8M |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 10 |

**Key Features:**
- Dense connectivity pattern for feature reuse
- Pretrained weights for transfer learning
- Efficient parameter usage

---

### 2. ResNet50 (`Model_ResNet50/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | ResNet-50 (pretrained on ImageNet) |
| **Parameters** | ~25M |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 10 |

**Key Features:**
- Residual connections for deeper networks
- Skip connections prevent gradient vanishing
- Standard baseline for medical imaging

---

### 3. DenseNet2 - Optimized (`Model_DenseNet2/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | DenseNet-121 with optimizations |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Focal Loss (γ=2.0) |
| **Threshold** | Class-specific optimal thresholds |
| **Epochs** | 10 |

**Key Features:**
- Focal Loss for handling class imbalance
- Per-class optimal threshold tuning
- Improved rare disease detection

---

### 4. Swin Transformer - Tiny (`Model_Swin/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | Swin-T (Shifted Window Transformer) |
| **Parameters** | ~28M |
| **Window Size** | 7 × 7 |
| **Optimizer** | AdamW (lr=0.0001) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 10 |

**Key Features:**
- Hierarchical vision transformer
- Shifted window attention mechanism
- Linear computational complexity

---

### 5. Microsoft Swin - Enhanced (`Model_microsoft_swin/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | swin_tiny_patch4_window7_224 |
| **Parameters** | ~28M |
| **Optimizer** | AdamW (lr=0.0001, weight_decay=0.05) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 20 |
| **Scheduler** | CosineAnnealingLR |

**Key Features:**
- Extended training (20 epochs)
- Cosine annealing learning rate schedule
- Best performing Swin variant
- **Best overall AUC: 0.8386**

---

### 6. ViT - Vision Transformer (`Model_ViT/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | ViT-B/16 (Base, patch 16×16) |
| **Parameters** | ~86M |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 30 |

**Key Features:**
- Pure transformer architecture
- Global self-attention mechanism
- Trained from scratch

---

### 7. ViT Pretrained (`Model_ViT_Pretrained/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | ViT-B/16 (ImageNet pretrained) |
| **Parameters** | ~86M |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 30 |

**Key Features:**
- Transfer learning from ImageNet
- Fine-tuned for chest X-rays
- Compared to scratch training

---

### 8. DeiT - Data-efficient Image Transformer (`model_deit_model/`)

| Specification | Value |
|---------------|-------|
| **Architecture** | deit_base_patch16_224 |
| **Parameters** | ~86M |
| **Optimizer** | AdamW (lr=0.0003) |
| **Loss** | Binary Cross-Entropy |
| **Epochs** | 10 |

**Key Features:**
- Knowledge distillation architecture
- Efficient training on smaller datasets
- Data augmentation strategies

---

### 9. DeiT v2 - Improved (`model_deit_2/`) ⭐ **BEST F1 SCORE**

| Specification | Value |
|---------------|-------|
| **Architecture** | deit_base_patch16_224 with enhanced head |
| **Parameters** | ~86M |
| **Optimizer** | AdamW (lr=0.0003, weight_decay=0.01) |
| **Loss** | Weighted Focal Loss (γ=2.0) |
| **Epochs** | 30 (3-phase training) |
| **Scheduler** | CosineAnnealingWarmRestarts |

**Key Improvements Applied:**
1. ✅ **Class Weights** in loss function (Hernia: 476×, Pneumonia: 79×)
2. ✅ **Optimal Threshold Tuning** (0.45 vs default 0.5)
3. ✅ **30 Epoch Training** with phase-based unfreezing
4. ✅ **CLAHE Augmentation** + rotation, flip, color jitter
5. ✅ **Gradual Backbone Unfreezing** (3 phases)
6. ✅ **Focal Loss** for rare disease detection
7. ✅ **Balanced Batch Sampling** with WeightedRandomSampler

**Training Phases:**
- Phase 1 (Epochs 1-5): Backbone frozen, train classifier only
- Phase 2 (Epochs 6-15): Last 4 transformer blocks unfrozen
- Phase 3 (Epochs 16-30): Entire network fine-tuned

---

## Performance Comparison

### Overall Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| DenseNet121 | 31.78% | 29.91% | 34.96% | 0.2916 | **0.8072** |
| ResNet50 | 31.24% | 30.05% | 34.92% | 0.2893 | 0.8023 |
| DenseNet2 | 31.05% | 28.94% | 25.44% | 0.2477 | 0.7799 |
| Swin-T | 31.46% | 29.67% | 33.66% | 0.2891 | 0.8036 |
| **Microsoft Swin** | **68.74%** | - | - | - | **0.8386** |
| ViT-B/16 | 28.97% | 27.33% | 29.31% | 0.2593 | 0.7766 |
| ViT Pretrained | 28.97% | 27.33% | 29.31% | 0.2593 | 0.7766 |
| DeiT Base | 88.66% | 36.80% | 48.49% | 0.4184 | 0.6897 |
| **DeiT v2** ⭐ | **86.41%** | **64.44%** | **74.25%** | **0.6734** | 0.7204 |

### Best Per-Class AUC Scores (Microsoft Swin)

| Disease | AUC Score |
|---------|-----------|
| Emphysema | 0.9340 |
| Hernia | 0.9214 |
| Edema | 0.9073 |
| Cardiomegaly | 0.9002 |
| Pneumothorax | 0.9010 |
| Effusion | 0.8765 |
| Mass | 0.8760 |

### Key Findings

1. **Best AUC**: Microsoft Swin Transformer (0.8386) - best for ranking predictions
2. **Best F1 Score**: DeiT v2 (0.6734) - best balance of precision/recall
3. **Best Recall**: DeiT v2 (74.25%) - critical for not missing diseases
4. **Most Improved**: DeiT v2 with +60% F1 improvement over baseline

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/Arjjun-S/CXR8.git
cd CXR8

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm pandas numpy matplotlib seaborn scikit-learn opencv-python tqdm pillow
```

### Dataset Download

Download the NIH ChestX-ray8 dataset from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC):

```bash
# Navigate to images folder
cd images

# Run download script
python batch_download_zips.py
```

---

## Usage

### Training a Model

```bash
# Train DenseNet121
python Model_DenseNet121/train.py

# Train ResNet50
python Model_ResNet50/train.py

# Train Microsoft Swin (best AUC)
python Model_microsoft_swin/train.py

# Train DeiT v2 (best F1)
python model_deit_2/train.py
```

### Generate Visualizations

```bash
# Generate graphs for any model
python Model_DenseNet121/generate_graphs.py
python model_deit_2/generate_graphs.py
```

### Data Preparation

```bash
# Prepare and organize the dataset
python prepare_data.py
```

---

## Project Structure

```
CXR8/
├── README.md                    # This file
├── prepare_data.py              # Dataset preparation script
├── run_all_models.py            # Run all model training
├── Data_Entry_2017_v2020.csv    # Main labels file
├── BBox_List_2017.csv           # Bounding box annotations
├── train_val_list.txt           # Train/val split list
├── test_list.txt                # Test split list
│
├── Model_DenseNet121/           # DenseNet-121 implementation
│   ├── train.py
│   ├── generate_graphs.py
│   ├── metrics.json
│   └── *.png                    # Generated visualizations
│
├── Model_ResNet50/              # ResNet-50 implementation
│   ├── train.py
│   ├── generate_graphs.py
│   └── metrics.json
│
├── Model_DenseNet2/             # Optimized DenseNet with Focal Loss
│   ├── train.py
│   └── metrics.json
│
├── Model_Swin/                  # Swin Transformer Tiny
│   ├── train.py
│   └── metrics.json
│
├── Model_microsoft_swin/        # Enhanced Swin (Best AUC)
│   ├── train.py
│   ├── generate_graphs.py
│   ├── eval_final.py
│   └── final_metrics.json
│
├── Model_ViT/                   # Vision Transformer from scratch
│   ├── train.py
│   └── metrics.json
│
├── Model_ViT_Pretrained/        # Pretrained ViT
│   ├── train.py
│   ├── continue_training.py
│   └── metrics.json
│
├── model_deit_model/            # DeiT Base
│   ├── train.py
│   ├── generate_graphs.py
│   └── final_metrics.json
│
├── model_deit_2/                # DeiT v2 - Improved (Best F1)
│   ├── train.py
│   ├── generate_graphs.py
│   ├── final_metrics.json
│   └── *.png                    # 12+ visualizations
│
├── Preprocessing/               # Data analysis scripts
│   ├── data_analysis.py
│   └── imbalance_report.txt
│
├── LongTailCXR/                 # Long-tail distribution labels
├── PruneCXR/                    # Pruned labels (MICCAI 2023)
├── MAPLEZ/                      # LLM-extracted improved labels
│
├── images/                      # Dataset images (not in repo)
│   ├── batch_download_zips.py
│   └── images_001/ to images_012/
│
└── train/                       # Organized training data
    └── [15 disease folders]/
```

---

## Results

### Confusion Matrices

Each model generates full 15×15 confusion matrices for detailed analysis:

- True vs Predicted labels for all disease classes
- Normalized and raw count versions
- Per-class accuracy visualization

### ROC Curves

Multi-class ROC curves with per-class AUC scores for each model.

### Training Curves

- Loss curves (train vs validation)
- Accuracy progression over epochs
- Learning rate schedules

---

## Key Observations

### What Works Best

1. **Transformer architectures** (Swin, DeiT) outperform CNNs on this dataset
2. **Class weighting** is crucial for handling the severe imbalance
3. **Focal Loss** significantly improves rare disease detection
4. **Phase-based training** prevents catastrophic forgetting
5. **CLAHE augmentation** enhances contrast for subtle findings

### Challenges

- **Infiltration** and **Pneumonia** remain difficult to distinguish
- **Hernia** detection benefits most from class weighting (476× weight)
- Trade-off between AUC (ranking quality) and F1 (classification quality)

---

## Future Work

- [ ] Ensemble of best models (Swin + DeiT v2)
- [ ] Attention visualization for interpretability
- [ ] Multi-task learning with bounding box localization
- [ ] Integration with LLM-based report generation

---

## References

1. [NIH ChestX-ray8 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. [DenseNet Paper](https://arxiv.org/abs/1608.06993)
3. [ResNet Paper](https://arxiv.org/abs/1512.03385)
4. [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
5. [DeiT Paper](https://arxiv.org/abs/2012.12877)
6. [Focal Loss Paper](https://arxiv.org/abs/1708.02002)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Arjjun S**

- GitHub: [@Arjjun-S](https://github.com/Arjjun-S)

---

## Acknowledgments

- NIH Clinical Center for the ChestX-ray8 dataset
- PyTorch and timm library teams
- Hugging Face for pretrained transformer weights

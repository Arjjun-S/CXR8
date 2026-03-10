"""
DeiT v2 (Improved) - Data-efficient Image Transformer for NIH CXR8
Multi-Label Classification with ENHANCED Training

IMPROVEMENTS:
1. Class Weights in Loss Function (pos_weight)
2. Optimal Threshold Tuning (F1-based)
3. 30 Epochs Training
4. Enhanced Data Augmentation (including CLAHE)
5. Phase-based Fine-tuning (gradual unfreezing)
6. Focal Loss for rare disease detection
7. Balanced Batch Sampling
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from datetime import datetime
import h5py
import warnings
import cv2
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, 
    recall_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Configuration
WORKSPACE = r"E:\arjjun\CXR8"
DATA_CSV = os.path.join(WORKSPACE, "Data_Entry_2017_v2020.csv")
IMAGES_BASE = os.path.join(WORKSPACE, "images")
OUTPUT_DIR = os.path.join(WORKSPACE, "model_deit_2")

# Training parameters
NUM_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224
NUM_WORKERS = 4
INITIAL_THRESHOLD = 0.5  # Will be optimized
WEIGHT_DECAY = 0.01

# Phase-based training configuration
PHASE1_EPOCHS = 5   # Freeze backbone, train classifier
PHASE2_EPOCHS = 10  # Unfreeze last layers
PHASE3_EPOCHS = 15  # Unfreeze entire network

# Focal Loss parameters
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Disease classes (15 total including No Finding)
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]
NUM_CLASSES = len(DISEASE_CLASSES)

# Style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ================== CUSTOM LOSS FUNCTIONS ==================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class WeightedFocalLoss(nn.Module):
    """Focal Loss with class weights"""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        return loss.mean()


# ================== MODEL ARCHITECTURE ==================

class ThoracicModelV2(nn.Module):
    """Enhanced DeiT backbone with improved classification head"""
    def __init__(self, model_name="deit_base_patch16_224", num_classes=15):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.head.in_features
        
        # Remove original head
        self.backbone.head = nn.Identity()
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen")
    
    def unfreeze_last_layers(self, num_blocks=4):
        """Unfreeze last N transformer blocks"""
        # Unfreeze last N blocks
        for block in self.backbone.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        # Unfreeze norm layer
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        print(f"  Last {num_blocks} blocks unfrozen")
    
    def unfreeze_all(self):
        """Unfreeze entire network"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  All layers unfrozen")


# ================== CUSTOM TRANSFORMS ==================

class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        if len(img_np.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = clahe.apply(img_np)
        
        return Image.fromarray(img_np)


# ================== DATASET ==================

def save_model_h5(model, filepath):
    """Save PyTorch model weights to HDF5 format"""
    try:
        state_dict = model.state_dict()
        with h5py.File(filepath, 'w') as f:
            for key, value in state_dict.items():
                f.create_dataset(key, data=value.cpu().numpy())
        print(f"  Model saved to {filepath}")
        return True
    except Exception as e:
        print(f"  Warning: Could not save .h5 file: {e}")
        return False


def find_image_path(image_name):
    """Search for image in images_001 through images_012 folders"""
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(IMAGES_BASE, folder, "images", image_name)
        if os.path.exists(path):
            return path
    return None


def create_label_vector(finding_labels):
    """Convert Finding Labels string to one-hot encoded vector"""
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    labels = finding_labels.split('|')
    for label in labels:
        label = label.strip()
        if label in DISEASE_CLASSES:
            idx = DISEASE_CLASSES.index(label)
            label_vector[idx] = 1.0
    return label_vector


class NIH_CXR8_Dataset(Dataset):
    """NIH Chest X-Ray dataset with enhanced features"""
    def __init__(self, dataframe, transform=None, return_weights=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_weights = return_weights
        self.image_paths = []
        self.labels = []
        
        print("Scanning for image paths...")
        valid_count = 0
        for idx in range(len(self.df)):
            img_name = self.df.iloc[idx]['Image Index']
            finding_labels = self.df.iloc[idx]['Finding Labels']
            
            img_path = find_image_path(img_name)
            if img_path:
                self.image_paths.append(img_path)
                self.labels.append(create_label_vector(finding_labels))
                valid_count += 1
            
            if (idx + 1) % 20000 == 0:
                print(f"  Processed {idx + 1}/{len(self.df)} entries, found {valid_count} images")
        
        self.labels = np.array(self.labels)
        print(f"Total valid images found: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        # Count positive samples per class
        pos_counts = self.labels.sum(axis=0)
        neg_counts = len(self.labels) - pos_counts
        
        # Calculate weights: total / (num_classes * class_count)
        # Higher weight for rare classes
        weights = neg_counts / (pos_counts + 1e-5)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self):
        """Calculate sample weights for balanced sampling"""
        # Weight each sample by inverse frequency of its rarest class
        class_counts = self.labels.sum(axis=0) + 1
        class_weights = 1.0 / class_counts
        
        sample_weights = []
        for label in self.labels:
            if label.sum() == 0:
                weight = 1.0
            else:
                weight = np.max(class_weights * label)
            sample_weights.append(weight)
        
        return np.array(sample_weights)


# ================== METRICS ==================

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Flatten for binary metrics
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
    specificity = tn / (tn + fp + 1e-6)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob, average='macro')
    except:
        roc_auc = 0.5
    
    return acc, precision, recall, specificity, f1, roc_auc


def compute_per_class_metrics(y_true, y_pred, y_prob):
    """Compute per-class metrics"""
    per_class_auc = []
    per_class_f1 = []
    per_class_ap = []
    
    for i in range(NUM_CLASSES):
        try:
            auc_score = roc_auc_score(y_true[:, i], y_prob[:, i])
        except:
            auc_score = 0.5
        per_class_auc.append(auc_score)
        
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        per_class_f1.append(f1)
        
        try:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
        except:
            ap = 0.0
        per_class_ap.append(ap)
    
    return per_class_auc, per_class_f1, per_class_ap


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find optimal threshold using validation data"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# ================== VISUALIZATION FUNCTIONS ==================

def plot_training_curves(history, output_dir):
    """Training vs Validation Loss and Accuracy Curves"""
    print("Generating Training Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
    # Mark phase transitions
    ax1.axvline(x=PHASE1_EPOCHS, color='green', linestyle='--', alpha=0.7, label='Phase 2 Start')
    ax1.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS, color='orange', linestyle='--', alpha=0.7, label='Phase 3 Start')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss (Phase-based)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
    ax2.axvline(x=PHASE1_EPOCHS, color='green', linestyle='--', alpha=0.7)
    ax2.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 and AUC curves
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_f1'], 'g-', linewidth=2, label='Validation F1', marker='o', markersize=3)
    ax3.plot(epochs, history['val_auc'], 'm-', linewidth=2, label='Validation AUC', marker='s', markersize=3)
    ax3.axvline(x=PHASE1_EPOCHS, color='green', linestyle='--', alpha=0.7)
    ax3.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS, color='orange', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('F1 Score and ROC-AUC Over Epochs', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate curve
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['lr'], 'c-', linewidth=2, marker='o', markersize=3)
    ax4.axvline(x=PHASE1_EPOCHS, color='green', linestyle='--', alpha=0.7, label='Phase 2')
    ax4.axvline(x=PHASE1_EPOCHS + PHASE2_EPOCHS, color='orange', linestyle='--', alpha=0.7, label='Phase 3')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_training_curves.png")


def plot_roc_curves(y_true, y_prob, output_dir):
    """ROC Curves (per class and macro average)"""
    print("Generating ROC Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    # First half of classes
    ax1 = axes[0, 0]
    for i in range(NUM_CLASSES // 2):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2, label=f'{DISEASE_CLASSES[i]} ({roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (Classes 1-7)')
    ax1.legend(loc='lower right', fontsize=8)
    
    # Second half of classes
    ax2 = axes[0, 1]
    for i in range(NUM_CLASSES // 2, NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=colors[i], lw=2, label=f'{DISEASE_CLASSES[i]} ({roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves (Classes 8-15)')
    ax2.legend(loc='lower right', fontsize=8)
    
    # Macro-average ROC
    ax3 = axes[1, 0]
    all_fpr = np.unique(np.concatenate([roc_curve(y_true[:, i], y_prob[:, i])[0] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax3.plot(all_fpr, mean_tpr, 'b-', lw=3, label=f'Macro-Average (AUC={macro_auc:.3f})')
    ax3.fill_between(all_fpr, mean_tpr, alpha=0.3)
    ax3.plot([0, 1], [0, 1], 'k--', lw=1)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Macro-Average ROC Curve')
    ax3.legend(loc='lower right')
    
    # Per-class AUC bar plot
    ax4 = axes[1, 1]
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    ax4.barh(range(NUM_CLASSES), sorted_aucs, color=plt.cm.RdYlGn(np.array(sorted_aucs)))
    ax4.set_yticks(range(NUM_CLASSES))
    ax4.set_yticklabels(sorted_names, fontsize=9)
    ax4.set_xlabel('AUC Score')
    ax4.set_title('Per-Class AUC Scores')
    ax4.set_xlim(0.5, 1.0)
    for i, v in enumerate(sorted_aucs):
        ax4.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_roc_curves.png")


def plot_precision_recall_curves(y_true, y_prob, output_dir):
    """Precision-Recall Curves"""
    print("Generating Precision-Recall Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    ax1 = axes[0]
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax1.plot(recall, precision, color=colors[i], lw=1.5, label=f'{DISEASE_CLASSES[i]} ({ap:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves')
    ax1.legend(loc='lower left', fontsize=7, ncol=2)
    
    ax2 = axes[1]
    aps = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aps)[::-1]
    sorted_aps = [aps[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    ax2.barh(range(NUM_CLASSES), sorted_aps, color=plt.cm.viridis(np.linspace(0.2, 0.8, NUM_CLASSES)))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel('Average Precision')
    ax2.set_title('Per-Class Average Precision')
    for i, v in enumerate(sorted_aps):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_precision_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_precision_recall.png")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Confusion Matrix (per class - top 4)"""
    print("Generating Confusion Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    top_classes = [0, 4, 8, 14]  # Atelectasis, Effusion, Infiltration, No Finding
    
    for idx, cls_idx in enumerate(top_classes):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true[:, cls_idx], y_pred[:, cls_idx])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix: {DISEASE_CLASSES[cls_idx]}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix.png")


def plot_confusion_matrix_full(y_true, y_pred, y_prob, output_dir):
    """Full 15x15 Confusion Matrix"""
    print("Generating Full Confusion Matrix...")
    
    # Convert multi-label to single-label
    y_true_single = []
    y_pred_single = []
    
    for i in range(len(y_true)):
        pred_class = np.argmax(y_prob[i])
        y_pred_single.append(pred_class)
        
        true_indices = np.where(y_true[i] == 1)[0]
        if len(true_indices) == 0:
            y_true_single.append(np.argmax(y_prob[i]))
        elif len(true_indices) == 1:
            y_true_single.append(true_indices[0])
        else:
            best_true = true_indices[np.argmax(y_prob[i, true_indices])]
            y_true_single.append(best_true)
    
    y_true_single = np.array(y_true_single)
    y_pred_single = np.array(y_pred_single)
    
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(NUM_CLASSES))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 7})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('DeiT v2 - Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Normalized
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 7})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('DeiT v2 - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full.png")


def plot_metric_comparison(metrics, output_dir):
    """Metric Comparison Bar Chart"""
    print("Generating Metric Comparison Chart...")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
              metrics['specificity'], metrics['f1'], metrics['roc_auc']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('DeiT v2 (Improved) - Performance Metrics', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_metric_comparison.png")


def plot_per_class_auc(y_true, y_prob, output_dir):
    """Per-Class AUC Bar Plot"""
    print("Generating Per-Class AUC Plot...")
    
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.array(sorted_aucs))
    ax.barh(range(NUM_CLASSES), sorted_aucs, color=colors)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('Per-Class ROC-AUC Scores (DeiT v2)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(sorted_aucs):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_auc.png")


def plot_per_class_f1(y_true, y_pred, output_dir):
    """Per-Class F1 Score Plot"""
    print("Generating Per-Class F1 Plot...")
    
    f1_scores = [f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, NUM_CLASSES))
    ax.barh(range(NUM_CLASSES), sorted_f1, color=colors)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores (DeiT v2)', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(sorted_f1):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_f1.png")


def plot_class_distribution(df, output_dir):
    """Class Distribution Plot"""
    print("Generating Class Distribution Plot...")
    
    label_counts = Counter()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            label = label.strip()
            if label in DISEASE_CLASSES:
                label_counts[label] += 1
    
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_labels]
    counts = [x[1] for x in sorted_labels]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(labels)))[::-1]
    bars = ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title('NIH CXR8 Class Distribution', fontsize=14, fontweight='bold')
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / len(df) * 100
        ax.text(count + 500, i, f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_class_distribution.png")


def plot_prediction_histogram(y_prob, threshold, output_dir):
    """Prediction Probability Histogram"""
    print("Generating Prediction Histogram...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(y_prob.ravel(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold={threshold:.2f}')
    ax1.axvline(x=0.5, color='orange', linestyle=':', linewidth=2, label='Default (0.5)')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Prediction Distribution')
    ax1.legend()
    
    top_classes = [14, 8, 4]  # No Finding, Infiltration, Effusion
    for idx, cls_idx in enumerate(top_classes):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        ax.hist(y_prob[:, cls_idx], bins=50, color=plt.cm.Set2(idx), edgecolor='black', alpha=0.7)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{DISEASE_CLASSES[cls_idx]} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_prediction_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_prediction_histogram.png")


def plot_threshold_analysis(y_true, y_prob, optimal_threshold, output_dir):
    """Plot threshold analysis"""
    print("Generating Threshold Analysis...")
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(int)
        p = precision_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        r = recall_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        f = f1_score(y_true.ravel(), y_pred.ravel(), average='macro', zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision', marker='o', markersize=4)
    ax.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall', marker='s', markersize=4)
    ax.plot(thresholds, f1_scores, 'r-', linewidth=2, label='F1 Score', marker='^', markersize=4)
    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Optimal={optimal_threshold:.2f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Default=0.5')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Analysis (Precision/Recall/F1)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_threshold_analysis.png")


def plot_class_weights(class_weights, output_dir):
    """Plot class weights visualization"""
    print("Generating Class Weights Plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_idx = np.argsort(class_weights.numpy())[::-1]
    sorted_weights = [class_weights[i].item() for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, NUM_CLASSES))
    ax.barh(range(NUM_CLASSES), sorted_weights, color=colors)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Class Weight (pos_weight)', fontsize=12)
    ax.set_title('Class Weights for Imbalanced Learning', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(sorted_weights):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_class_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_class_weights.png")


def save_metrics_txt(metrics, history, output_dir, best_epoch, optimal_threshold, class_weights):
    """Save metrics to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    content = f"""======================================================================
MODEL PERFORMANCE METRICS - DeiT v2 (IMPROVED)
======================================================================

Model: DeiT Base (deit_base_patch16_224) with Enhanced Head
Timestamp: {timestamp}
Device: GPU (NVIDIA)

----------------------------------------------------------------------
IMPROVEMENTS APPLIED
----------------------------------------------------------------------
  1. Class Weights in Loss Function (pos_weight)
  2. Optimal Threshold Tuning (F1-based): {optimal_threshold:.2f}
  3. 30 Epochs Training (Phase-based)
  4. Enhanced Data Augmentation (CLAHE, rotation, flip)
  5. Gradual Backbone Unfreezing (3 phases)
  6. Focal Loss for rare disease detection
  7. Balanced Batch Sampling

----------------------------------------------------------------------
TRAINING PHASES
----------------------------------------------------------------------
  Phase 1 (Epochs 1-{PHASE1_EPOCHS}):     Backbone frozen, train classifier
  Phase 2 (Epochs {PHASE1_EPOCHS+1}-{PHASE1_EPOCHS+PHASE2_EPOCHS}):    Last 4 blocks unfrozen
  Phase 3 (Epochs {PHASE1_EPOCHS+PHASE2_EPOCHS+1}-{NUM_EPOCHS}):   Entire network unfrozen

----------------------------------------------------------------------
HYPERPARAMETERS
----------------------------------------------------------------------
  Epochs:          {NUM_EPOCHS}
  Batch size:      {BATCH_SIZE}
  Learning rate:   {LEARNING_RATE}
  Weight decay:    {WEIGHT_DECAY}
  Optimizer:       AdamW
  Scheduler:       CosineAnnealingWarmRestarts
  Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}
  Optimal Threshold: {optimal_threshold:.2f}
  Loss:            Focal Loss (gamma={FOCAL_GAMMA})

----------------------------------------------------------------------
ARCHITECTURE
----------------------------------------------------------------------
  Backbone:        DeiT Base (ViT) - deit_base_patch16_224
  Head:            Linear(768->1024) -> BN -> GELU -> Dropout(0.3)
                   -> Linear(1024->512) -> BN -> GELU -> Dropout(0.3)
                   -> Linear(512->15)

----------------------------------------------------------------------
DATASET (FULL NIH CXR8)
----------------------------------------------------------------------
  Total images:    112,120
  Train images:    ~89,696
  Val images:      ~22,424

----------------------------------------------------------------------
CLASS WEIGHTS (pos_weight)
----------------------------------------------------------------------
"""
    
    for i in range(NUM_CLASSES):
        content += f"  {DISEASE_CLASSES[i]:<20}: {class_weights[i].item():.2f}\n"
    
    content += f"""
----------------------------------------------------------------------
FINAL METRICS (Best Model - Epoch {best_epoch})
----------------------------------------------------------------------
  Accuracy                 : {metrics['accuracy']:.4f}
  Precision                : {metrics['precision']:.4f}
  Recall (Sensitivity)     : {metrics['recall']:.4f}
  Specificity              : {metrics['specificity']:.4f}
  F1-Score                 : {metrics['f1']:.4f}
  ROC-AUC                  : {metrics['roc_auc']:.4f}

  Optimal Threshold        : {optimal_threshold:.2f}
  Best Validation AUC      : {metrics['roc_auc']:.4f}

----------------------------------------------------------------------
PER-CLASS ROC-AUC SCORES
----------------------------------------------------------------------
"""
    
    # Add per-class AUC
    for i, auc_val in enumerate(metrics['per_class_auc']):
        content += f"  {DISEASE_CLASSES[i]:<20}: {auc_val:.4f}\n"
    
    content += """
----------------------------------------------------------------------
SAVED FILES
----------------------------------------------------------------------
  model.h5                 : HDF5 format weights
  model.pth                : PyTorch weights only
  best_model.pth           : Best model checkpoint
  final_model.pth          : Final epoch model
  metrics.txt              : This file
  metrics.json             : JSON format metrics
  final_metrics.json       : Final metrics JSON
  graph_*.png              : All visualizations (12+ graphs)

======================================================================
"""
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(content)
    
    print(f"  Saved: metrics.txt")


# ================== MAIN TRAINING FUNCTION ==================

def train():
    """Main training function with all improvements"""
    print("="*70)
    print("DeiT v2 (IMPROVED) TRAINING FOR NIH CXR8 MULTI-LABEL CLASSIFICATION")
    print("="*70)
    print("\nIMPROVEMENTS:")
    print("  1. Class Weights in Loss Function")
    print("  2. Optimal Threshold Tuning")
    print("  3. 30 Epochs Training")
    print("  4. Enhanced Data Augmentation (CLAHE)")
    print("  5. Phase-based Backbone Fine-tuning")
    print("  6. Focal Loss")
    print("  7. Balanced Batch Sampling")
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_CSV)
    print(f"Total entries: {len(df)}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training entries: {len(train_df)}")
    print(f"Validation entries: {len(val_df)}")
    
    # Enhanced transforms with CLAHE
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHETransform(clip_limit=2.0),  # CLAHE for contrast
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # ±10 degrees
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("\nPreparing training dataset...")
    train_dataset = NIH_CXR8_Dataset(train_df, transform=train_transform)
    print("\nPreparing validation dataset...")
    val_dataset = NIH_CXR8_Dataset(val_df, transform=val_transform)
    
    # Calculate class weights for imbalanced data
    print("\nCalculating class weights...")
    class_weights = train_dataset.get_class_weights().to(device)
    print("Class weights calculated:")
    for i, w in enumerate(class_weights):
        print(f"  {DISEASE_CLASSES[i]:<20}: {w.item():.2f}")
    
    # Balanced sampling
    print("\nSetting up balanced batch sampling...")
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Plot class distribution and weights
    plot_class_distribution(df, OUTPUT_DIR)
    plot_class_weights(class_weights.cpu(), OUTPUT_DIR)
    
    # Create model
    print("\nCreating ThoracicModelV2 (Enhanced DeiT)...")
    model = ThoracicModelV2(model_name="deit_base_patch16_224", num_classes=NUM_CLASSES)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Focal Loss with class weights
    criterion = WeightedFocalLoss(gamma=FOCAL_GAMMA, pos_weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [],
        'lr': [], 'phase': []
    }
    
    best_auc = 0
    best_f1 = 0
    best_epoch = 0
    best_metrics = None
    optimal_threshold = INITIAL_THRESHOLD
    
    print("\n" + "="*70)
    print("STARTING PHASE-BASED TRAINING")
    print("="*70)
    
    try:
        for epoch in range(NUM_EPOCHS):
            # Determine current phase and adjust model
            if epoch == 0:
                print(f"\n{'='*70}")
                print("PHASE 1: Freezing backbone, training classifier only")
                print(f"{'='*70}")
                model.freeze_backbone()
                # Reset optimizer for new phase
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
                current_phase = 1
            
            elif epoch == PHASE1_EPOCHS:
                print(f"\n{'='*70}")
                print("PHASE 2: Unfreezing last 4 transformer blocks")
                print(f"{'='*70}")
                model.unfreeze_last_layers(num_blocks=4)
                # Reset optimizer with lower LR
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
                current_phase = 2
            
            elif epoch == PHASE1_EPOCHS + PHASE2_EPOCHS:
                print(f"\n{'='*70}")
                print("PHASE 3: Unfreezing entire network")
                print(f"{'='*70}")
                model.unfreeze_all()
                # Reset optimizer with even lower LR
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
                current_phase = 3
            
            history['phase'].append(current_phase)
            
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy with current threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > optimal_threshold).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
                
                if (batch_idx + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} [Phase {current_phase}] - "
                          f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # Record learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            y_true, y_pred, y_prob = [], [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    
                    loss = criterion(outputs, labels.to(device))
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > optimal_threshold).astype(int)
                    
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds)
                    y_prob.extend(probs)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Find optimal threshold every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                optimal_threshold, best_f1_at_thresh = find_optimal_threshold(y_true, y_prob, metric='f1')
                print(f"  Updated optimal threshold: {optimal_threshold:.2f} (F1: {best_f1_at_thresh:.4f})")
                # Recalculate predictions with new threshold
                y_pred = (y_prob > optimal_threshold).astype(int)
            
            # Compute metrics
            acc, prec, rec, spec, f1, roc_auc = compute_metrics(y_true, y_pred, y_prob)
            per_class_auc, per_class_f1, _ = compute_per_class_metrics(y_true, y_pred, y_prob)
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(acc)
            history['val_auc'].append(roc_auc)
            history['val_f1'].append(f1)
            history['val_precision'].append(prec)
            history['val_recall'].append(rec)
            
            # Print epoch results
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} [Phase {current_phase}]")
            print(f"{'='*70}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Accuracy:   {acc:.4f}")
            print(f"  Precision:  {prec:.4f}")
            print(f"  Recall:     {rec:.4f}")
            print(f"  Specificity:{spec:.4f}")
            print(f"  F1 Score:   {f1:.4f}")
            print(f"  ROC AUC:    {roc_auc:.4f}")
            print(f"  Threshold:  {optimal_threshold:.2f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Save best model based on F1 score (better for imbalanced data)
            if f1 > best_f1:
                best_f1 = f1
                best_auc = roc_auc
                best_epoch = epoch + 1
                best_metrics = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'specificity': spec,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'per_class_auc': per_class_auc,
                    'per_class_f1': per_class_f1,
                    'optimal_threshold': optimal_threshold
                }
                
                # Save best model checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'best_auc': best_auc,
                    'optimal_threshold': optimal_threshold,
                    'metrics': best_metrics
                }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                
                print(f"\n  *** NEW BEST MODEL SAVED! F1: {best_f1:.4f}, AUC: {best_auc:.4f} ***")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"  Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
    
    except KeyboardInterrupt:
        print("\n\n*** Training interrupted by user ***")
        print("Saving current best model...")
    
    except Exception as e:
        print(f"\n\n*** Training error: {e} ***")
        import traceback
        traceback.print_exc()
        print("Saving current best model...")
    
    finally:
        # Always save models at the end
        print("\n" + "="*70)
        print("SAVING FINAL MODELS")
        print("="*70)
        
        # Save final model (PyTorch)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
        print("  Saved: final_model.pth")
        
        # Save as model.pth
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))
        print("  Saved: model.pth")
        
        # Save as .h5
        save_model_h5(model, os.path.join(OUTPUT_DIR, 'model.h5'))
        
        # Save training history
        with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print("  Saved: training_history.json")
        
        # If we have best metrics, generate visualizations
        if best_metrics:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
            
            # Load best model for evaluation
            checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'),
                                   map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
            model.eval()
            
            # Get final predictions on validation set
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > optimal_threshold).astype(int)
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds)
                    y_prob.extend(probs)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            
            # Generate all visualizations
            plot_training_curves(history, OUTPUT_DIR)
            plot_roc_curves(y_true, y_prob, OUTPUT_DIR)
            plot_precision_recall_curves(y_true, y_prob, OUTPUT_DIR)
            plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)
            plot_confusion_matrix_full(y_true, y_pred, y_prob, OUTPUT_DIR)
            plot_metric_comparison(best_metrics, OUTPUT_DIR)
            plot_per_class_auc(y_true, y_prob, OUTPUT_DIR)
            plot_per_class_f1(y_true, y_pred, OUTPUT_DIR)
            plot_prediction_histogram(y_prob, optimal_threshold, OUTPUT_DIR)
            plot_threshold_analysis(y_true, y_prob, optimal_threshold, OUTPUT_DIR)
            
            # Save metrics
            save_metrics_txt(best_metrics, history, OUTPUT_DIR, best_epoch, 
                           optimal_threshold, class_weights.cpu())
            
            # Save metrics.json
            metrics_json = {
                'model': 'deit_base_patch16_224_v2',
                'improvements': [
                    'Class Weights',
                    'Optimal Threshold Tuning',
                    'Phase-based Training',
                    'Focal Loss',
                    'CLAHE Augmentation',
                    'Balanced Sampling'
                ],
                'best_epoch': best_epoch,
                'best_val_f1': best_f1,
                'best_val_auc': best_auc,
                'optimal_threshold': optimal_threshold,
                'metrics': {k: v for k, v in best_metrics.items() 
                           if k not in ['per_class_auc', 'per_class_f1']},
                'class_aucs': {DISEASE_CLASSES[i]: best_metrics['per_class_auc'][i] 
                              for i in range(NUM_CLASSES)},
                'class_f1s': {DISEASE_CLASSES[i]: best_metrics['per_class_f1'][i] 
                             for i in range(NUM_CLASSES)},
                'class_weights': {DISEASE_CLASSES[i]: class_weights[i].item() 
                                 for i in range(NUM_CLASSES)}
            }
            with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print("  Saved: metrics.json")
            
            # Save final_metrics.json
            with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print("  Saved: final_metrics.json")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest Model: Epoch {best_epoch}")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Best ROC-AUC: {best_auc:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        print(f"\nAll files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

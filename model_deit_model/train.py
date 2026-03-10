"""
DeiT (Data-efficient Image Transformer) for NIH CXR8 Multi-Label Classification
Uses the FULL dataset (112,120 images) from images_001 to images_012
30 epochs with comprehensive metrics and visualization logging

ThoracicModel: DeiT backbone with custom classification head
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
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
OUTPUT_DIR = os.path.join(WORKSPACE, "model_deit_model")

# Training parameters
NUM_EPOCHS = 30
BATCH_SIZE = 16  # Smaller batch for GPU memory
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224
NUM_WORKERS = 4
THRESHOLD = 0.5
WEIGHT_DECAY = 0.01

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


class ThoracicModel(nn.Module):
    """DeiT backbone with custom classification head for multi-label learning"""
    def __init__(self, model_name="deit_base_patch16_224", num_classes=15):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True)
        in_features = self.backbone.head.in_features
        
        # Remove original head
        self.backbone.head = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


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
    """NIH Chest X-Ray dataset using full CSV with image search"""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
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
        
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        per_class_ap.append(ap)
    
    return per_class_auc, per_class_f1, per_class_ap


# ================== VISUALIZATION FUNCTIONS ==================

def plot_training_curves(history, output_dir):
    """1 & 2. Training vs Validation Loss and Accuracy Curves"""
    print("Generating Training Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_training_curves.png")


def plot_roc_curves(y_true, y_prob, output_dir):
    """3. ROC Curves (per class and macro average)"""
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
    """4. Precision-Recall Curves"""
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
    """5. Confusion Matrix (per class)"""
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


def plot_metric_comparison(metrics, output_dir):
    """6. Metric Comparison Bar Chart"""
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
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_metric_comparison.png")


def plot_per_class_auc(y_true, y_prob, output_dir):
    """7. Per-Class AUC Bar Plot"""
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
    ax.set_title('Per-Class ROC-AUC Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(sorted_aucs):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_auc.png")


def plot_per_class_f1(y_true, y_pred, output_dir):
    """8. Per-Class F1 Score Plot"""
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
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(sorted_f1):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_f1.png")


def plot_class_distribution(df, output_dir):
    """9. Class Distribution Plot"""
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


def plot_learning_rate_curve(history, output_dir):
    """10. Learning Rate vs Epoch Curve"""
    print("Generating Learning Rate Curve...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['lr']) + 1)
    
    ax.plot(epochs, history['lr'], 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (CosineAnnealingLR)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_learning_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_learning_rate.png")


def plot_prediction_histogram(y_prob, output_dir):
    """12. Prediction Probability Histogram"""
    print("Generating Prediction Histogram...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(y_prob.ravel(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Prediction Distribution')
    ax1.legend()
    
    top_classes = [14, 8, 4]  # No Finding, Infiltration, Effusion
    for idx, cls_idx in enumerate(top_classes):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        ax.hist(y_prob[:, cls_idx], bins=50, color=plt.cm.Set2(idx), edgecolor='black', alpha=0.7)
        ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{DISEASE_CLASSES[cls_idx]} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_prediction_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_prediction_histogram.png")


def generate_gradcam_visualization(model, dataloader, device, output_dir, num_samples=6):
    """11. Grad-CAM Visualizations"""
    print("Generating Grad-CAM Visualizations...")
    
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  Skipping Grad-CAM (pytorch-grad-cam not installed)")
        print("  Install with: pip install pytorch-grad-cam")
        return
    
    model.eval()
    
    # Get target layer - last block of DeiT backbone
    target_layers = [model.backbone.blocks[-1].norm1]
    
    try:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=lambda x: x[:, 1:, :].reshape(x.shape[0], 14, 14, -1).permute(0, 3, 1, 2))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Get sample images
        sample_images = []
        sample_labels = []
        
        for images, labels in dataloader:
            for i in range(len(images)):
                if len(sample_images) >= num_samples:
                    break
                sample_images.append(images[i])
                sample_labels.append(labels[i])
            if len(sample_images) >= num_samples:
                break
        
        for idx in range(num_samples):
            img_tensor = sample_images[idx].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                pred_class = torch.argmax(output).item()
            
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Denormalize image
            img_np = sample_images[idx].permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            cam_image = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
            
            ax = axes[idx]
            ax.imshow(cam_image)
            ax.set_title(f'Pred: {DISEASE_CLASSES[pred_class]}', fontsize=10)
            ax.axis('off')
        
        plt.suptitle('Grad-CAM Visualizations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'graph_gradcam.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: graph_gradcam.png")
    except Exception as e:
        print(f"  Grad-CAM failed: {e}")


def save_metrics_txt(metrics, history, output_dir, best_epoch):
    """Save metrics to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    content = f"""======================================================================
MODEL PERFORMANCE METRICS
======================================================================

Model: DeiT Base (deit_base_patch16_224) with Custom Head
Timestamp: {timestamp}
Device: GPU (NVIDIA)

----------------------------------------------------------------------
HYPERPARAMETERS
----------------------------------------------------------------------
  Epochs:          {NUM_EPOCHS}
  Batch size:      {BATCH_SIZE}
  Learning rate:   {LEARNING_RATE}
  Weight decay:    {WEIGHT_DECAY}
  Optimizer:       AdamW
  Scheduler:       CosineAnnealingLR
  Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}
  Threshold:       {THRESHOLD}
  Loss:            BCEWithLogitsLoss

----------------------------------------------------------------------
ARCHITECTURE
----------------------------------------------------------------------
  Backbone:        DeiT Base (ViT) - deit_base_patch16_224
  Head:            Linear(768->512) -> BN -> ReLU -> Dropout(0.4) -> Linear(512->15)

----------------------------------------------------------------------
DATASET (FULL NIH CXR8)
----------------------------------------------------------------------
  Total images:    112,120
  Train images:    ~89,696
  Val images:      ~22,424

----------------------------------------------------------------------
FINAL METRICS (Best Model - Epoch {best_epoch})
----------------------------------------------------------------------
  Accuracy                 : {metrics['accuracy']:.4f}
  Precision                : {metrics['precision']:.4f}
  Recall (Sensitivity)     : {metrics['recall']:.4f}
  Specificity              : {metrics['specificity']:.4f}
  F1-Score                 : {metrics['f1']:.4f}
  ROC-AUC                  : {metrics['roc_auc']:.4f}

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
  graph_*.png              : All visualizations

======================================================================
"""
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(content)
    
    print(f"  Saved: metrics.txt")


def train():
    """Main training function"""
    print("="*60)
    print("DeiT TRAINING FOR NIH CXR8 MULTI-LABEL CLASSIFICATION")
    print("="*60)
    
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
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Plot class distribution
    plot_class_distribution(df, OUTPUT_DIR)
    
    # Create model
    print("\nCreating ThoracicModel (DeiT backbone)...")
    model = ThoracicModel(model_name="deit_base_patch16_224", num_classes=NUM_CLASSES)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'lr': []
    }
    
    best_auc = 0
    best_epoch = 0
    best_metrics = None
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        for epoch in range(NUM_EPOCHS):
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
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                probs = torch.sigmoid(outputs)
                preds = (probs > THRESHOLD).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()
                
                if (batch_idx + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # Record learning rate
            current_lr = scheduler.get_last_lr()[0]
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
                    preds = (probs > THRESHOLD).astype(int)
                    
                    y_true.extend(labels.numpy())
                    y_pred.extend(preds)
                    y_prob.extend(probs)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Compute metrics
            acc, prec, rec, spec, f1, roc_auc = compute_metrics(y_true, y_pred, y_prob)
            per_class_auc, per_class_f1, _ = compute_per_class_metrics(y_true, y_pred, y_prob)
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(acc)
            history['val_auc'].append(roc_auc)
            
            # Print epoch results
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"{'='*60}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  Specificity: {spec:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if roc_auc > best_auc:
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
                    'per_class_f1': per_class_f1
                }
                
                # Save best model checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'metrics': best_metrics
                }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                
                print(f"\n  *** NEW BEST MODEL SAVED! AUC: {best_auc:.4f} ***")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"  Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
    
    except KeyboardInterrupt:
        print("\n\n*** Training interrupted by user ***")
        print("Saving current best model...")
    
    except Exception as e:
        print(f"\n\n*** Training error: {e} ***")
        print("Saving current best model...")
    
    finally:
        # Always save models at the end
        print("\n" + "="*60)
        print("SAVING FINAL MODELS")
        print("="*60)
        
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
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS")
            print("="*60)
            
            # Load best model for evaluation
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), 
                                             map_location=device, weights_only=False)['model_state_dict'])
            model.eval()
            
            # Get final predictions on validation set
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > THRESHOLD).astype(int)
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
            plot_metric_comparison(best_metrics, OUTPUT_DIR)
            plot_per_class_auc(y_true, y_prob, OUTPUT_DIR)
            plot_per_class_f1(y_true, y_pred, OUTPUT_DIR)
            plot_learning_rate_curve(history, OUTPUT_DIR)
            plot_prediction_histogram(y_prob, OUTPUT_DIR)
            
            # Grad-CAM
            generate_gradcam_visualization(model, val_loader, device, OUTPUT_DIR)
            
            # Save metrics
            save_metrics_txt(best_metrics, history, OUTPUT_DIR, best_epoch)
            
            # Save metrics.json
            metrics_json = {
                'model': 'deit_base_patch16_224',
                'best_epoch': best_epoch,
                'best_val_auc': best_auc,
                'metrics': best_metrics,
                'class_aucs': {DISEASE_CLASSES[i]: best_metrics['per_class_auc'][i] for i in range(NUM_CLASSES)}
            }
            with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print("  Saved: metrics.json")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nBest Model: Epoch {best_epoch}")
        print(f"Best ROC-AUC: {best_auc:.4f}")
        print(f"\nAll files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

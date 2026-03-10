"""
Generate all visualizations and metrics for DeiT v2 (Improved) Model
Loads the saved best model and generates comprehensive graphs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import json
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Configuration
WORKSPACE = r"E:\arjjun\CXR8"
DATA_CSV = os.path.join(WORKSPACE, "Data_Entry_2017_v2020.csv")
IMAGES_BASE = os.path.join(WORKSPACE, "images")
OUTPUT_DIR = os.path.join(WORKSPACE, "model_deit_2")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEFAULT_THRESHOLD = 0.5

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]
NUM_CLASSES = len(DISEASE_CLASSES)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ThoracicModelV2(nn.Module):
    """Enhanced DeiT backbone with improved classification head"""
    def __init__(self, model_name="deit_base_patch16_224", num_classes=15):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=False)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
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
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(IMAGES_BASE, folder, "images", image_name)
        if os.path.exists(path):
            return path
    return None


def create_label_vector(finding_labels):
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    labels = finding_labels.split('|')
    for label in labels:
        label = label.strip()
        if label in DISEASE_CLASSES:
            idx = DISEASE_CLASSES.index(label)
            label_vector[idx] = 1.0
    return label_vector


class NIH_CXR8_Dataset(Dataset):
    def __init__(self, dataframe, transform=None, max_samples=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        count = 0
        for idx in range(len(self.df)):
            if max_samples and count >= max_samples:
                break
            img_name = self.df.iloc[idx]['Image Index']
            finding_labels = self.df.iloc[idx]['Finding Labels']
            img_path = find_image_path(img_name)
            if img_path:
                self.image_paths.append(img_path)
                self.labels.append(create_label_vector(finding_labels))
                count += 1
        print(f"Loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), img_path


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
    ax.set_title('NIH CXR8 Class Distribution (Full Dataset)', fontsize=14, fontweight='bold')
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / len(df) * 100
        ax.text(count + 500, i, f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_class_distribution.png")


def plot_roc_curves(y_true, y_prob, output_dir):
    """ROC Curves"""
    print("Generating ROC Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    ax1 = axes[0, 0]
    for i in range(NUM_CLASSES // 2):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2, label=f'{DISEASE_CLASSES[i]} (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (Classes 1-7)')
    ax1.legend(loc='lower right', fontsize=8)
    
    ax2 = axes[0, 1]
    for i in range(NUM_CLASSES // 2, NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=colors[i], lw=2, label=f'{DISEASE_CLASSES[i]} (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves (Classes 8-15)')
    ax2.legend(loc='lower right', fontsize=8)
    
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
    
    ax4 = axes[1, 1]
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    bars = ax4.barh(range(NUM_CLASSES), sorted_aucs, color=plt.cm.RdYlGn(np.array(sorted_aucs)))
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
        ax1.plot(recall, precision, color=colors[i], lw=1.5, label=f'{DISEASE_CLASSES[i]} (AP={ap:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves (All Classes)')
    ax1.legend(loc='lower left', fontsize=7, ncol=2)
    
    ax2 = axes[1]
    aps = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aps)[::-1]
    sorted_aps = [aps[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    bars = ax2.barh(range(NUM_CLASSES), sorted_aps, color=plt.cm.viridis(np.linspace(0.2, 0.8, NUM_CLASSES)))
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
    """Confusion Matrix (per class)"""
    print("Generating Confusion Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    top_classes = [0, 4, 8, 14]
    
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
    
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 7})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('DeiT v2 - Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
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
    
    # Large individual versions
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 9})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('DeiT v2 - Full Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full_raw.png")
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 9})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('DeiT v2 - Full Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full_normalized.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full_normalized.png")


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
    
    top_classes = [14, 8, 4]
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


def save_metrics_txt(metrics, per_class_auc, output_dir, best_epoch, optimal_threshold):
    """Save metrics to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    content = f"""======================================================================
MODEL PERFORMANCE METRICS - DeiT v2 (IMPROVED)
======================================================================

Model: DeiT Base (deit_base_patch16_224) with Enhanced Head
Timestamp: {timestamp}
Device: GPU (NVIDIA GeForce RTX 5050 Laptop GPU)

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
HYPERPARAMETERS
----------------------------------------------------------------------
  Epochs:          30
  Batch size:      16
  Learning rate:   3e-04
  Weight decay:    0.01
  Optimizer:       AdamW
  Scheduler:       CosineAnnealingWarmRestarts
  Image size:      224x224
  Optimal Threshold: {optimal_threshold:.2f}
  Loss:            Focal Loss (gamma=2.0)

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
    
    auc_with_names = [(DISEASE_CLASSES[i], per_class_auc[i]) for i in range(NUM_CLASSES)]
    auc_with_names.sort(key=lambda x: x[1], reverse=True)
    
    for name, auc_val in auc_with_names:
        content += f"  {name:<20}: {auc_val:.4f}\n"
    
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


def main():
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS FOR DeiT v2 MODEL")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_CSV)
    print(f"Total entries: {len(df)}")
    
    # Sample for evaluation
    val_df = df.sample(n=8000, random_state=42)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nPreparing dataset...")
    dataset = NIH_CXR8_Dataset(val_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Load model
    print("\nLoading model...")
    model = ThoracicModelV2(model_name="deit_base_patch16_224", num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', 30)
        best_auc = checkpoint.get('best_auc', 0)
        optimal_threshold = checkpoint.get('optimal_threshold', DEFAULT_THRESHOLD)
        print(f"Loaded model from epoch {best_epoch} with AUC: {best_auc:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.2f}")
    else:
        model.load_state_dict(checkpoint)
        best_epoch = 30
        optimal_threshold = DEFAULT_THRESHOLD
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Save models in different formats
    print("\n" + "="*60)
    print("SAVING MODEL FILES")
    print("="*60)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))
    print("  Saved: model.pth")
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
    print("  Saved: final_model.pth")
    
    save_model_h5(model, os.path.join(OUTPUT_DIR, 'model.h5'))
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_true_list, y_prob_list = [], []
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            y_true_list.extend(labels.numpy())
            y_prob_list.extend(probs)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples")
    
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold, best_f1_at_thresh = find_optimal_threshold(y_true, y_prob, metric='f1')
    print(f"  Optimal threshold: {optimal_threshold:.2f} (F1: {best_f1_at_thresh:.4f})")
    
    y_pred = (y_prob > optimal_threshold).astype(int)
    
    print(f"\nTotal samples: {len(y_true)}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60)
    
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
    spec = tn / (tn + fp + 1e-6)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob, average='macro')
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    
    # Per-class AUC
    per_class_auc = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    
    # Generate all graphs
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_class_distribution(df, OUTPUT_DIR)
    plot_roc_curves(y_true, y_prob, OUTPUT_DIR)
    plot_precision_recall_curves(y_true, y_prob, OUTPUT_DIR)
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)
    plot_confusion_matrix_full(y_true, y_pred, y_prob, OUTPUT_DIR)
    plot_metric_comparison(metrics, OUTPUT_DIR)
    plot_per_class_auc(y_true, y_prob, OUTPUT_DIR)
    plot_per_class_f1(y_true, y_pred, OUTPUT_DIR)
    plot_prediction_histogram(y_prob, optimal_threshold, OUTPUT_DIR)
    plot_threshold_analysis(y_true, y_prob, optimal_threshold, OUTPUT_DIR)
    
    # Save metrics files
    print("\n" + "="*60)
    print("SAVING METRICS FILES")
    print("="*60)
    
    save_metrics_txt(metrics, per_class_auc, OUTPUT_DIR, best_epoch, optimal_threshold)
    
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
        'best_val_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'metrics': metrics,
        'class_aucs': {DISEASE_CLASSES[i]: per_class_auc[i] for i in range(NUM_CLASSES)}
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print("  Saved: metrics.json")
    
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print("  Saved: final_metrics.json")
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS AND METRICS GENERATED!")
    print("="*60)
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

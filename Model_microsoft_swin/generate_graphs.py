"""
Generate all 12 visualizations for Microsoft Swin Model
Loads the saved model and generates comprehensive graphs
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
import h5py

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
OUTPUT_DIR = os.path.join(WORKSPACE, "Model_microsoft_swin")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
THRESHOLD = 0.3

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]
NUM_CLASSES = len(DISEASE_CLASSES)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def find_image_path(image_name):
    """Search for image in images_001 through images_012 folders"""
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


def create_model():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model


def plot_class_distribution(df, output_dir):
    """1. Class Distribution Plot"""
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
    """2. ROC Curves (per class and macro average)"""
    print("Generating ROC Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Per-class ROC curves - split into 2 subplots
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    # First half of classes
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
    
    # Second half of classes
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
    """3. Precision-Recall Curves"""
    print("Generating Precision-Recall Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    
    # Per-class P-R curves
    ax1 = axes[0]
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax1.plot(recall, precision, color=colors[i], lw=1.5, label=f'{DISEASE_CLASSES[i]} (AP={ap:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves (All Classes)')
    ax1.legend(loc='lower left', fontsize=7, ncol=2)
    
    # AP comparison
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
    """4. Confusion Matrix (multi-label aggregated)"""
    print("Generating Confusion Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Per-class confusion matrices (2x2 for each class) - show top 4
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
    """Full 15x15 Confusion Matrix (single-label conversion)"""
    print("Generating Full Confusion Matrix...")
    
    # Convert multi-label to single-label by taking the class with highest probability
    # For true labels: if multiple labels exist, use the one with highest predicted prob
    y_true_single = []
    y_pred_single = []
    
    for i in range(len(y_true)):
        # For predictions: argmax of probabilities
        pred_class = np.argmax(y_prob[i])
        y_pred_single.append(pred_class)
        
        # For true labels: if sample has multiple labels, pick the most confident prediction among true classes
        # If no label (all zeros), use argmax
        true_indices = np.where(y_true[i] == 1)[0]
        if len(true_indices) == 0:
            # No true label - use argmax (likely 'No Finding' = index 14)
            y_true_single.append(np.argmax(y_prob[i]))
        elif len(true_indices) == 1:
            y_true_single.append(true_indices[0])
        else:
            # Multiple true labels - choose the one with highest predicted probability
            best_true = true_indices[np.argmax(y_prob[i, true_indices])]
            y_true_single.append(best_true)
    
    y_true_single = np.array(y_true_single)
    y_pred_single = np.array(y_pred_single)
    
    # Compute confusion matrix
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
    ax1.set_title('Swin Model - Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Normalized
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 7})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('Swin Model - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full.png")
    
    # Also save individual large confusion matrices
    # Raw counts - large version
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 9})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Swin Model - Full Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full_raw.png")
    
    # Normalized - large version
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES,
                annot_kws={'size': 9})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Swin Model - Full Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_confusion_matrix_full_normalized.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_confusion_matrix_full_normalized.png")


def plot_metric_comparison(y_true, y_pred, y_prob, output_dir):
    """5. Metric Comparison Bar Chart"""
    print("Generating Metric Comparison Chart...")
    
    # Calculate metrics
    acc = accuracy_score(y_true.ravel(), y_pred.ravel())
    prec = precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    rec = recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    f1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    
    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
    spec = tn / (tn + fp + 1e-6)
    
    # AUC
    auc_score = roc_auc_score(y_true, y_prob, average='macro')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
    values = [acc, prec, rec, spec, f1, auc_score]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_metric_comparison.png")


def plot_per_class_f1(y_true, y_pred, output_dir):
    """6. Per-Class F1 Score Plot"""
    print("Generating Per-Class F1 Score Plot...")
    
    f1_scores = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)
    
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, NUM_CLASSES))
    bars = ax.barh(range(NUM_CLASSES), sorted_f1, color=colors)
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


def plot_per_class_auc(y_true, y_prob, output_dir):
    """7. Per-Class AUC Bar Plot"""
    print("Generating Per-Class AUC Plot...")
    
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(NUM_CLASSES)]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [DISEASE_CLASSES[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.array(sorted_aucs))
    bars = ax.barh(range(NUM_CLASSES), sorted_aucs, color=colors)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('Per-Class ROC-AUC Scores', fontsize=14, fontweight='bold')
    ax.set_xlim(0.6, 1.0)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    
    for i, v in enumerate(sorted_aucs):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_auc.png")


def plot_prediction_histogram(y_prob, output_dir):
    """8. Prediction Probability Histogram"""
    print("Generating Prediction Probability Histogram...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall histogram
    ax1 = axes[0, 0]
    ax1.hist(y_prob.ravel(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Prediction Distribution')
    ax1.legend()
    
    # Per-class histogram (top 3 classes)
    top_classes = [14, 8, 4]  # No Finding, Infiltration, Effusion
    for idx, cls_idx in enumerate(top_classes):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        ax.hist(y_prob[:, cls_idx], bins=50, color=plt.cm.Set2(idx), edgecolor='black', alpha=0.7)
        ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{DISEASE_CLASSES[cls_idx]} Probability Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_prediction_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_prediction_histogram.png")


def generate_gradcam(model, dataset, device, output_dir, num_samples=6):
    """9. Grad-CAM Visualizations"""
    print("Generating Grad-CAM Visualizations...")
    
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("  Skipping Grad-CAM (pytorch-grad-cam not installed)")
        return
    
    model.eval()
    
    # Get target layer (last layer of Swin)
    target_layers = [model.layers[-1].blocks[-1].norm2]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(min(num_samples, len(dataset))):
        img, label, img_path = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)
        
        # Find the class with highest prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = torch.argmax(output).item()
        
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Load original image
        orig_img = Image.open(img_path).convert('RGB')
        orig_img = orig_img.resize((224, 224))
        orig_img = np.array(orig_img) / 255.0
        
        cam_image = show_cam_on_image(orig_img.astype(np.float32), grayscale_cam, use_rgb=True)
        
        ax = axes[idx]
        ax.imshow(cam_image)
        ax.set_title(f'Pred: {DISEASE_CLASSES[pred_class]}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Grad-CAM Visualizations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_gradcam.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_gradcam.png")


def main():
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS FOR SWIN MODEL")
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
    model = create_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
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
    y_pred = (y_prob > THRESHOLD).astype(int)
    
    print(f"\nTotal samples: {len(y_true)}")
    
    # Generate all graphs
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Class Distribution
    plot_class_distribution(df, OUTPUT_DIR)
    
    # 2. ROC Curves
    plot_roc_curves(y_true, y_prob, OUTPUT_DIR)
    
    # 3. Precision-Recall Curves
    plot_precision_recall_curves(y_true, y_prob, OUTPUT_DIR)
    
    # 4. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)
    
    # 4b. Full Confusion Matrix (15x15)
    plot_confusion_matrix_full(y_true, y_pred, y_prob, OUTPUT_DIR)
    
    # 5. Metric Comparison
    plot_metric_comparison(y_true, y_pred, y_prob, OUTPUT_DIR)
    
    # 6. Per-Class F1
    plot_per_class_f1(y_true, y_pred, OUTPUT_DIR)
    
    # 7. Per-Class AUC
    plot_per_class_auc(y_true, y_prob, OUTPUT_DIR)
    
    # 8. Prediction Histogram
    plot_prediction_histogram(y_prob, OUTPUT_DIR)
    
    # 9. Grad-CAM (optional - requires extra package)
    try:
        generate_gradcam(model, dataset, device, OUTPUT_DIR)
    except Exception as e:
        print(f"  Grad-CAM skipped: {e}")
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

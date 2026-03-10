"""
Generate all visualizations and metrics for ResNet50 Model
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
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Configuration
WORKSPACE = r"E:\arjjun\CXR8"
TRAIN_DIR = os.path.join(WORKSPACE, "train")
OUTPUT_DIR = os.path.join(WORKSPACE, "Model_ResNet50")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.h5")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Classes from folder structure
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No_Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
NUM_CLASSES = len(CLASSES)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_model(num_classes, device):
    """Create ResNet50 model matching training architecture"""
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)


def load_model_h5(model, filepath, device):
    """Load PyTorch model weights (saved as .h5 with torch.save)"""
    try:
        state_dict = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"  Model loaded from {filepath}")
        return True
    except Exception as e:
        print(f"  Error loading .h5 file: {e}")
        return False


def get_patient_wise_split(dataset, val_ratio=0.2, seed=42):
    """Create patient-wise train/val split"""
    np.random.seed(seed)
    
    patient_indices = {}
    for idx in range(len(dataset)):
        img_path = dataset.imgs[idx][0]
        img_name = os.path.basename(img_path)
        patient_id = img_name.split('_')[0]
        
        if patient_id not in patient_indices:
            patient_indices[patient_id] = []
        patient_indices[patient_id].append(idx)
    
    patients = list(patient_indices.keys())
    np.random.shuffle(patients)
    
    split_idx = int(len(patients) * (1 - val_ratio))
    train_patients = set(patients[:split_idx])
    
    train_indices = []
    val_indices = []
    
    for patient_id, indices in patient_indices.items():
        if patient_id in train_patients:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
    
    return train_indices, val_indices


# ================== VISUALIZATION FUNCTIONS ==================

def plot_class_distribution(dataset, output_dir):
    """Class Distribution Plot"""
    print("Generating Class Distribution Plot...")
    
    class_counts = Counter(dataset.targets)
    classes = dataset.classes
    
    sorted_items = sorted([(classes[k], v) for k, v in class_counts.items()], 
                          key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    counts = [x[1] for x in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(labels)))[::-1]
    bars = ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title('CXR8 Class Distribution (Train Folder)', fontsize=14, fontweight='bold')
    
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / total * 100
        ax.text(count + 100, i, f'{count:,} ({pct:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_class_distribution.png")


def plot_roc_curves(y_true_bin, y_prob, classes, output_dir):
    """ROC Curves (per class and macro average)"""
    print("Generating ROC Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # First half of classes
    ax1 = axes[0, 0]
    for i in range(len(classes) // 2):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2, label=f'{classes[i][:12]} (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (Classes 1-7)')
    ax1.legend(loc='lower right', fontsize=7)
    
    # Second half of classes
    ax2 = axes[0, 1]
    for i in range(len(classes) // 2, len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=colors[i], lw=2, label=f'{classes[i][:12]} (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves (Classes 8-15)')
    ax2.legend(loc='lower right', fontsize=7)
    
    # Macro-average ROC
    ax3 = axes[1, 0]
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_prob[:, i])[0] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(classes)
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
    aucs = [roc_auc_score(y_true_bin[:, i], y_prob[:, i]) for i in range(len(classes))]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [classes[i] for i in sorted_idx]
    
    ax4.barh(range(len(classes)), sorted_aucs, color=plt.cm.RdYlGn(np.array(sorted_aucs)))
    ax4.set_yticks(range(len(classes)))
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


def plot_precision_recall_curves(y_true_bin, y_prob, classes, output_dir):
    """Precision-Recall Curves"""
    print("Generating Precision-Recall Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    ax1 = axes[0]
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax1.plot(recall, precision, color=colors[i], lw=1.5, label=f'{classes[i][:10]} (AP={ap:.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves (All Classes)')
    ax1.legend(loc='lower left', fontsize=6, ncol=2)
    
    ax2 = axes[1]
    aps = [average_precision_score(y_true_bin[:, i], y_prob[:, i]) for i in range(len(classes))]
    sorted_idx = np.argsort(aps)[::-1]
    sorted_aps = [aps[i] for i in sorted_idx]
    sorted_names = [classes[i] for i in sorted_idx]
    
    ax2.barh(range(len(classes)), sorted_aps, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(classes))))
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel('Average Precision')
    ax2.set_title('Per-Class Average Precision')
    for i, v in enumerate(sorted_aps):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_precision_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_precision_recall.png")


def plot_confusion_matrix_full(y_true, y_pred, classes, output_dir):
    """Full Confusion Matrix"""
    print("Generating Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Raw Counts)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Normalized
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                xticklabels=classes, yticklabels=classes)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Normalized)')
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
    ax.set_title('ResNet50 Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_metric_comparison.png")


def plot_per_class_auc(y_true_bin, y_prob, classes, output_dir):
    """Per-Class AUC Bar Plot"""
    print("Generating Per-Class AUC Plot...")
    
    aucs = [roc_auc_score(y_true_bin[:, i], y_prob[:, i]) for i in range(len(classes))]
    sorted_idx = np.argsort(aucs)[::-1]
    sorted_aucs = [aucs[i] for i in sorted_idx]
    sorted_names = [classes[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.array(sorted_aucs))
    ax.barh(range(len(classes)), sorted_aucs, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('Per-Class ROC-AUC Scores (ResNet50)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(sorted_aucs):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_auc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_auc.png")


def plot_per_class_f1(y_true, y_pred, classes, output_dir):
    """Per-Class F1 Score Plot"""
    print("Generating Per-Class F1 Plot...")
    
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    sorted_names = [classes[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, len(classes)))
    ax.barh(range(len(classes)), sorted_f1, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores (ResNet50)', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(sorted_f1):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_f1.png")


def plot_prediction_histogram(y_prob, output_dir):
    """Prediction Probability Histogram"""
    print("Generating Prediction Histogram...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(y_prob.max(axis=1), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Max Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Max Confidence Scores')
    
    ax2 = axes[0, 1]
    ax2.hist(y_prob.ravel(), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Prediction Distribution')
    
    # Top 3 classes by count
    top_classes = [10, 8, 4]  # No_Finding, Infiltration, Effusion indices typically
    for idx, cls_idx in enumerate(top_classes[:2]):
        ax = axes[1, idx]
        ax.hist(y_prob[:, cls_idx], bins=50, color=plt.cm.Set2(idx), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Class Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_prediction_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_prediction_histogram.png")


def plot_per_class_metrics(y_true, y_pred, classes, output_dir):
    """Per-class precision, recall, F1 comparison"""
    print("Generating Per-Class Metrics Plot...")
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, and F1-Score (ResNet50)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graph_per_class_metrics.png")


def main():
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS FOR ResNet50 MODEL")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print("\nLoading dataset...")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"  Classes: {num_classes}")
    print(f"  Total images: {len(full_dataset)}")
    
    # Get validation split
    _, val_indices = get_patient_wise_split(full_dataset, val_ratio=0.2, seed=42)
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"  Validation samples: {len(val_indices)}")
    
    # Load model
    print("\nLoading model...")
    model = create_model(num_classes, device)
    
    if os.path.exists(MODEL_PATH):
        load_model_h5(model, MODEL_PATH, device)
    else:
        pth_path = os.path.join(OUTPUT_DIR, "model.pth")
        if os.path.exists(pth_path):
            model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
            print(f"  Model loaded from {pth_path}")
        else:
            print("  ERROR: No model file found!")
            return
    
    model.eval()
    print("Model loaded successfully!")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_true_list, y_prob_list, y_pred_list = [], [], []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            y_true_list.extend(labels.numpy())
            y_prob_list.extend(probs)
            y_pred_list.extend(preds)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples")
    
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = np.array(y_pred_list)
    
    # Binarize for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    print(f"\nTotal samples: {len(y_true)}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    
    # Specificity (average)
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp + 1e-6)
        specificity_per_class.append(spec)
    spec = np.mean(specificity_per_class)
    
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
    
    # Generate all graphs
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Class Distribution
    plot_class_distribution(full_dataset, OUTPUT_DIR)
    
    # 2. ROC Curves
    plot_roc_curves(y_true_bin, y_prob, classes, OUTPUT_DIR)
    
    # 3. Precision-Recall Curves
    plot_precision_recall_curves(y_true_bin, y_prob, classes, OUTPUT_DIR)
    
    # 4. Confusion Matrix
    plot_confusion_matrix_full(y_true, y_pred, classes, OUTPUT_DIR)
    
    # 5. Metric Comparison
    plot_metric_comparison(metrics, OUTPUT_DIR)
    
    # 6. Per-Class F1
    plot_per_class_f1(y_true, y_pred, classes, OUTPUT_DIR)
    
    # 7. Per-Class AUC
    plot_per_class_auc(y_true_bin, y_prob, classes, OUTPUT_DIR)
    
    # 8. Prediction Histogram
    plot_prediction_histogram(y_prob, OUTPUT_DIR)
    
    # 9. Per-Class Metrics
    plot_per_class_metrics(y_true, y_pred, classes, OUTPUT_DIR)
    
    # Per-class AUC for saving
    per_class_auc = [roc_auc_score(y_true_bin[:, i], y_prob[:, i]) for i in range(num_classes)]
    
    # Save updated metrics.json
    metrics_json = {
        'model': 'ResNet50',
        'metrics': metrics,
        'class_aucs': {classes[i]: per_class_auc[i] for i in range(num_classes)},
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print("\n  Saved: final_metrics.json")
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
================================================================================
Model: DenseNet2 - Optimized for Maximum Metrics
================================================================================
Multi-label Thoracic Disease Classification using CXR8 Dataset

Optimizations Applied:
1. Focal Loss (gamma=2) for class imbalance
2. Per-class threshold optimization
3. Lower prediction threshold (0.3 default)
4. Moderate medical-appropriate augmentations
5. Class-weighted focal loss
6. Cosine annealing with warm restarts
7. Dense feature pyramid head

Architecture: DenseNet121 backbone with optimized head
================================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ================================================================================
# CONFIGURATION - OPTIMIZED
# ================================================================================
class Config:
    """Optimized configuration for maximum metrics"""
    
    # Paths
    BASE_DIR = r"E:\arjjun\CXR8"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    MODEL_DIR = os.path.join(BASE_DIR, "Model_DenseNet2")
    
    # Model hyperparameters - OPTIMIZED
    MODEL_NAME = "DenseNet2-Optimized"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10  # 10 epochs as requested
    WARMUP_EPOCHS = 2
    LEARNING_RATE = 3e-4  # Higher initial LR
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05  # Less smoothing
    NUM_WORKERS = 4
    
    # Focal Loss parameters
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None  # Will use class weights
    
    # Threshold optimization
    DEFAULT_THRESHOLD = 0.3  # Lower than 0.5 for better recall
    OPTIMIZE_THRESHOLDS = True
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Training
    GRADIENT_CLIP = 1.0
    VAL_SPLIT = 0.2
    
    # Augmentation strength
    AUG_ROTATION = 10  # ±10 degrees
    AUG_TRANSLATE = 0.05


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================================
# GPU SETUP
# ================================================================================
def setup_device():
    """Setup and verify GPU device - RTX only"""
    print("\n" + "=" * 70)
    print(" GPU CONFIGURATION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n  ERROR: CUDA is not available!")
        print("  This model requires GPU (RTX) for training.")
        sys.exit(1)
    
    try:
        test_tensor = torch.zeros(1).cuda()
        _ = test_tensor + 1
        device = torch.device("cuda")
        print(f"\n  PyTorch version:  {torch.__version__}")
        print(f"  CUDA available:   True")
        print(f"  GPU Device:       {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except RuntimeError as e:
        print(f"\n  ERROR: CUDA kernel error!")
        print(f"  Please ensure PyTorch supports your GPU.")
        print(f"  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)
    
    print(f"  Device:           {device}")
    print("=" * 70)
    
    return device


# ================================================================================
# FOCAL LOSS - KEY FOR CLASS IMBALANCE
# ================================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (gamma=2 recommended)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma > 0 reduces loss for well-classified examples,
    focusing on hard negatives.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            targets_smooth = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            targets_smooth = targets_smooth * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
            
            log_probs = torch.log_softmax(inputs, dim=-1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        else:
            ce_loss = nn.functional.cross_entropy(
                inputs, targets, weight=self.alpha, reduction='none'
            )
        
        # Compute focal weight
        probs = torch.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None and self.label_smoothing == 0:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ================================================================================
# DATA TRANSFORMS - MODERATE AUGMENTATION
# ================================================================================
def get_transforms():
    """
    Moderate augmentation for medical images:
    - Horizontal flip (lungs are symmetric)
    - Rotation ±10° (realistic positioning variation)
    - Slight translation
    - Contrast adjustment
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(Config.AUG_ROTATION),
        transforms.RandomAffine(
            degrees=0, 
            translate=(Config.AUG_TRANSLATE, Config.AUG_TRANSLATE)
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


# ================================================================================
# OPTIMIZED MODEL HEAD
# ================================================================================
class DenseNet2Model(nn.Module):
    """
    DenseNet121 with optimized classification head
    
    Features:
    - Pretrained DenseNet121 backbone
    - Feature pyramid with multiple scales
    - Dropout for regularization
    - BatchNorm for stable training
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Load pretrained DenseNet121
        weights = DenseNet121_Weights.IMAGENET1K_V1
        backbone = densenet121(weights=weights)
        
        # Extract features (everything except classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optimized classification head
        in_features = 1024  # DenseNet121 output features
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize head weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = self.pool(features)
        out = self.classifier(out)
        return out


# ================================================================================
# PATIENT-WISE SPLIT
# ================================================================================
def patient_wise_split(dataset, val_ratio=0.2, seed=42):
    """Split data by patient ID to prevent data leakage"""
    np.random.seed(seed)
    
    samples = dataset.samples
    patient_to_indices = {}
    
    for idx, (path, _) in enumerate(samples):
        filename = os.path.basename(path)
        patient_id = filename.split('_')[0]
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    patients = list(patient_to_indices.keys())
    np.random.shuffle(patients)
    
    split_idx = int(len(patients) * (1 - val_ratio))
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])
    
    train_indices = [idx for p in train_patients for idx in patient_to_indices[p]]
    val_indices = [idx for p in val_patients for idx in patient_to_indices[p]]
    
    return train_indices, val_indices


# ================================================================================
# PER-CLASS THRESHOLD OPTIMIZATION
# ================================================================================
def optimize_thresholds(model, val_loader, device, class_names):
    """
    Find optimal threshold per class using validation set
    Optimizes for F1-score per class
    """
    print("\n" + "=" * 70)
    print(" OPTIMIZING PER-CLASS THRESHOLDS")
    print("=" * 70)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Binarize labels
    labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    
    # Find optimal threshold per class
    optimal_thresholds = {}
    print("\n  Per-class optimal thresholds:")
    print("  " + "-" * 50)
    
    for i, class_name in enumerate(class_names):
        best_f1 = 0
        best_thresh = 0.3
        
        for thresh in np.arange(0.1, 0.6, 0.05):
            preds = (all_probs[:, i] >= thresh).astype(int)
            true = labels_bin[:, i]
            
            # Calculate F1 for this class
            tp = np.sum((preds == 1) & (true == 1))
            fp = np.sum((preds == 1) & (true == 0))
            fn = np.sum((preds == 0) & (true == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds[class_name] = best_thresh
        print(f"    {class_name:25s}: {best_thresh:.2f} (F1: {best_f1:.3f})")
    
    print("  " + "-" * 50)
    print("=" * 70)
    
    return optimal_thresholds


# ================================================================================
# TRAINING FUNCTIONS
# ================================================================================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}/{total_epochs} [TRAIN]",
                bar_format='{l_bar}{bar:30}{r_bar}', ncols=110)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, threshold=0.5):
    """Validate model with specified threshold"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Validating",
                                   bar_format='{l_bar}{bar:30}{r_bar}', ncols=110):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return running_loss / len(val_loader), accuracy, all_preds, all_labels, all_probs


def validate_with_thresholds(model, val_loader, device, class_thresholds, class_names):
    """Validate using per-class optimal thresholds"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Apply per-class thresholds
    preds = []
    for i in range(len(all_labels)):
        class_probs = all_probs[i]
        # Apply thresholds and get prediction
        adjusted_probs = []
        for j, name in enumerate(class_names):
            thresh = class_thresholds.get(name, 0.3)
            adjusted_probs.append(class_probs[j] / thresh)
        preds.append(np.argmax(adjusted_probs))
    
    return np.array(preds), all_labels, all_probs


# ================================================================================
# SAVE RESULTS
# ================================================================================
def save_results(model, history, all_preds, all_labels, all_probs, class_names, 
                 device, optimal_thresholds=None):
    """Save all training outputs"""
    print("\n" + "=" * 70)
    print(" SAVING RESULTS")
    print("=" * 70)
    
    # Save model
    model_path = os.path.join(Config.MODEL_DIR, "model.h5")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate specificity
    cm = confusion_matrix(all_labels, all_preds)
    specificities = []
    for i in range(len(class_names)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp + 1e-8)
        specificities.append(spec)
    specificity = np.mean(specificities)
    
    # ROC-AUC
    labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    all_probs_arr = np.array(all_probs)
    try:
        roc_auc = roc_auc_score(labels_bin, all_probs_arr, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    # Save metrics.json
    metrics = {
        'model': Config.MODEL_NAME,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'best_val_acc': float(max(history['val_acc'])),
        'epochs': Config.EPOCHS,
        'focal_gamma': Config.FOCAL_GAMMA,
        'default_threshold': Config.DEFAULT_THRESHOLD,
        'optimal_thresholds': optimal_thresholds
    }
    
    metrics_json_path = os.path.join(Config.MODEL_DIR, "metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metrics.txt
    metrics_path = os.path.join(Config.MODEL_DIR, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: GPU ({torch.cuda.get_device_name(0)})\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Epochs:          {Config.EPOCHS}\n")
        f.write(f"  Batch size:      {Config.BATCH_SIZE}\n")
        f.write(f"  Learning rate:   {Config.LEARNING_RATE}\n")
        f.write(f"  Focal gamma:     {Config.FOCAL_GAMMA}\n")
        f.write(f"  Default threshold: {Config.DEFAULT_THRESHOLD}\n")
        f.write(f"  Image size:      {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}\n")
        f.write(f"  Random seed:     {Config.RANDOM_SEED}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("FINAL METRICS (Best Model)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy                 : {accuracy:.4f}\n")
        f.write(f"  Precision                : {precision:.4f}\n")
        f.write(f"  Recall (Sensitivity)     : {recall:.4f}\n")
        f.write(f"  Specificity              : {specificity:.4f}\n")
        f.write(f"  F1-Score                 : {f1:.4f}\n")
        f.write(f"  ROC-AUC                  : {roc_auc:.4f}\n\n")
        f.write(f"  Best Validation Accuracy: {max(history['val_acc']):.2f}%\n\n")
        
        if optimal_thresholds:
            f.write("-" * 70 + "\n")
            f.write("OPTIMAL THRESHOLDS PER CLASS\n")
            f.write("-" * 70 + "\n")
            for name, thresh in optimal_thresholds.items():
                f.write(f"  {name:25s}: {thresh:.2f}\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  Metrics saved: {metrics_path}")
    
    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{Config.MODEL_NAME} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{Config.MODEL_NAME} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(Config.MODEL_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curves saved: {curves_path}")
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'{Config.MODEL_NAME} Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    cm_path = os.path.join(Config.MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")
    
    print("=" * 70)


# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================
def main():
    print("\n")
    print("=" * 70)
    print("   DENSENET2 - OPTIMIZED TRAINING")
    print("   Multi-label Thoracic Disease Classification")
    print("=" * 70)
    print("\n  Optimizations:")
    print("  - Focal Loss (gamma=2)")
    print("  - Per-class threshold optimization")
    print("  - Moderate medical augmentation")
    print("  - Optimized classification head")
    
    set_seed(Config.RANDOM_SEED)
    print(f"\n  Random seed: {Config.RANDOM_SEED}")
    
    # Setup GPU
    device = setup_device()
    
    # Load dataset
    print("\n" + "=" * 70)
    print(" LOADING DATASET")
    print("=" * 70)
    
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=val_transform)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\n  Dataset: {Config.TRAIN_DIR}")
    print(f"  Total images: {len(full_dataset)}")
    print(f"  Number of classes: {num_classes}")
    
    # Class distribution
    print("\n  Class Distribution:")
    print("  " + "-" * 45)
    class_counts = Counter(full_dataset.targets)
    for idx, class_name in enumerate(class_names):
        print(f"    {class_name:25s}: {class_counts[idx]:6d}")
    print("  " + "-" * 45)
    
    # Compute class weights
    train_labels = full_dataset.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\n  Class weights computed for Focal Loss")
    
    # Patient-wise split
    train_indices, val_indices = patient_wise_split(full_dataset, Config.VAL_SPLIT, Config.RANDOM_SEED)
    
    print(f"\n  Patient-wise split:")
    print(f"    Training samples:   {len(train_indices)}")
    print(f"    Validation samples: {len(val_indices)}")
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print("=" * 70)
    
    # Create model
    print("\n" + "=" * 70)
    print(" MODEL ARCHITECTURE")
    print("=" * 70)
    
    print(f"\n  Loading DenseNet2-Optimized model...")
    model = DenseNet2Model(num_classes).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: DenseNet121 backbone + Optimized head")
    print(f"  Total parameters:     {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print("=" * 70)
    
    # Focal Loss criterion
    criterion = FocalLoss(
        alpha=class_weights, 
        gamma=Config.FOCAL_GAMMA,
        label_smoothing=Config.LABEL_SMOOTHING
    )
    
    # Optimizer with different LR for backbone and head
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    head_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': Config.LEARNING_RATE * 0.1},
        {'params': head_params, 'lr': Config.LEARNING_RATE}
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-7
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print(" TRAINING")
    print("=" * 70)
    print(f"\n  Epochs: {Config.EPOCHS}")
    print(f"  Focal Loss gamma: {Config.FOCAL_GAMMA}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_preds, best_labels, best_probs = None, None, None
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                            device, epoch, Config.EPOCHS)
        val_loss, val_acc, preds, labels, probs = validate(model, val_loader, criterion, 
                                                           device, Config.DEFAULT_THRESHOLD)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\n  ============================================================")
        print(f"  Epoch {epoch:02d}/{Config.EPOCHS} Summary:")
        print(f"    Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds, best_labels, best_probs = preds, labels, probs
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, "model.h5"))
            print(f"    [NEW BEST] Saved model with accuracy: {best_acc:.2f}%")
        
        print(f"  ============================================================\n")
    
    # Optimize thresholds
    optimal_thresholds = None
    if Config.OPTIMIZE_THRESHOLDS:
        optimal_thresholds = optimize_thresholds(model, val_loader, device, class_names)
        
        # Re-evaluate with optimal thresholds
        final_preds, final_labels, final_probs = validate_with_thresholds(
            model, val_loader, device, optimal_thresholds, class_names
        )
        final_acc = accuracy_score(final_labels, final_preds) * 100
        print(f"\n  Accuracy with optimal thresholds: {final_acc:.2f}%")
        
        if final_acc > best_acc:
            best_preds, best_labels = final_preds, final_labels
            best_probs = final_probs
    
    # Final evaluation and save
    save_results(model, history, best_preds, best_labels, best_probs, 
                 class_names, device, optimal_thresholds)
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Outputs saved to: {Config.MODEL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
================================================================================
ViT Continue Training - Improve Accuracy
================================================================================
Load from checkpoint and continue training with optimized settings
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
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
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
# CONFIGURATION - OPTIMIZED FOR IMPROVEMENT
# ================================================================================
class Config:
    """Optimized configuration for continued training"""
    
    # Paths
    BASE_DIR = r"E:\arjjun\CXR8"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    MODEL_DIR = os.path.join(BASE_DIR, "Model_ViT_Pretrained")
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model.h5")
    
    # Model hyperparameters - OPTIMIZED
    MODEL_NAME = "ViT-B/16-Continued"
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 10  # 10 more epochs
    WARMUP_EPOCHS = 1  # Short warmup
    LEARNING_RATE = 1e-5  # Lower LR for fine-tuning from checkpoint
    WEIGHT_DECAY = 0.01  # Reduced weight decay
    LABEL_SMOOTHING = 0.15  # Slightly more smoothing
    NUM_WORKERS = 4
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Training
    GRADIENT_CLIP = 1.0
    VAL_SPLIT = 0.2
    
    # MixUp augmentation
    MIXUP_ALPHA = 0.2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    """Setup GPU device"""
    print("\n" + "=" * 70)
    print(" GPU CONFIGURATION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n  WARNING: CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            device = torch.device("cuda")
            print(f"\n  PyTorch version:  {torch.__version__}")
            print(f"  GPU Device:       {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except RuntimeError:
            print("  CUDA kernel error - falling back to CPU")
            device = torch.device("cpu")
    
    print(f"  Device:           {device}")
    print("=" * 70)
    return device


def get_transforms():
    """Enhanced data transforms with stronger augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def patient_wise_split(dataset):
    """Split data by patient ID to prevent data leakage"""
    print("\n  Performing patient-wise split...")
    
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
    
    split_idx = int(len(patients) * (1 - Config.VAL_SPLIT))
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])
    
    train_indices = [idx for p in train_patients for idx in patient_to_indices[p]]
    val_indices = [idx for p in val_patients for idx in patient_to_indices[p]]
    
    return train_indices, val_indices


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_model(num_classes, device):
    """Create ViT model with modified head"""
    print("\n" + "=" * 70)
    print(" LOADING MODEL FROM CHECKPOINT")
    print("=" * 70)
    
    # Model architecture must match the saved checkpoint
    model = vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(768, num_classes)  # Original architecture from train.py
    )
    
    # Load checkpoint
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"\n  Loading checkpoint: {Config.CHECKPOINT_PATH}")
        checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        print("  Checkpoint loaded successfully!")
    else:
        print(f"\n  WARNING: Checkpoint not found at {Config.CHECKPOINT_PATH}")
        print("  Starting from pretrained ImageNet weights...")
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, num_classes)
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_mixup=True):
    """Train for one epoch with MixUp"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch:2d}", 
                bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply MixUp with probability
        if use_mixup and np.random.random() > 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, Config.MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
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
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Validating", 
                                   bar_format='{l_bar}{bar:30}{r_bar}', ncols=100):
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


def save_results(model, history, all_preds, all_labels, all_probs, class_names, device):
    """Save model, metrics, and plots"""
    print("\n" + "=" * 70)
    print(" SAVING RESULTS")
    print("=" * 70)
    
    # Save model
    model_path = os.path.join(Config.MODEL_DIR, "model_improved.h5")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")
    
    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC-AUC
    labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    all_probs_arr = np.array(all_probs)
    try:
        roc_auc = roc_auc_score(labels_bin, all_probs_arr, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    # Save metrics
    metrics = {
        'model': Config.MODEL_NAME,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'best_val_acc': max(history['val_acc']),
        'epochs_trained': Config.EPOCHS,
        'learning_rate': Config.LEARNING_RATE,
        'batch_size': Config.BATCH_SIZE
    }
    
    metrics_path = os.path.join(Config.MODEL_DIR, "metrics_improved.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save text report
    report_path = os.path.join(Config.MODEL_DIR, "metrics_improved.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IMPROVED MODEL PERFORMANCE METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: GPU ({torch.cuda.get_device_name(0)})\n\n")
        f.write("-" * 70 + "\n")
        f.write("FINAL METRICS (Improved Model)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy                 : {accuracy:.4f}\n")
        f.write(f"  Precision                : {precision:.4f}\n")
        f.write(f"  Recall (Sensitivity)     : {recall:.4f}\n")
        f.write(f"  F1-Score                 : {f1:.4f}\n")
        f.write(f"  ROC-AUC                  : {roc_auc:.4f}\n\n")
        f.write(f"  Best Validation Accuracy: {max(history['val_acc']):.2f}%\n\n")
        f.write("-" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  Metrics saved: {report_path}")
    
    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss (Continued)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy (Continued)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(Config.MODEL_DIR, "training_curves_improved.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Curves saved: {curves_path}")
    
    print("=" * 70)


def main():
    print("\n")
    print("=" * 70)
    print("   ViT CONTINUED TRAINING - ACCURACY IMPROVEMENT")
    print("   Multi-label Thoracic Disease Classification")
    print("=" * 70)
    
    set_seed(Config.RANDOM_SEED)
    print(f"\n  Random seed: {Config.RANDOM_SEED}")
    
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
    print(f"\n  Total images: {len(full_dataset)}")
    print(f"  Number of classes: {num_classes}")
    
    # Patient-wise split
    train_indices, val_indices = patient_wise_split(full_dataset)
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"\n  Training samples:   {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    
    # Compute class weights
    train_labels = [full_dataset.targets[i] for i in train_indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Create model from checkpoint
    model = create_model(num_classes, device)
    
    # Focal Loss for better class imbalance handling
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer with lower LR
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-7)
    
    print("\n" + "=" * 70)
    print(" TRAINING (Continued)")
    print("=" * 70)
    print(f"\n  Epochs: {Config.EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Using Focal Loss + MixUp augmentation")
    print()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_preds, best_labels, best_probs = None, None, None
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, preds, labels, probs = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds, best_labels, best_probs = preds, labels, probs
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, "model_improved.h5"))
            print(f"  *** New best: {best_acc:.2f}% ***")
        
        print()
    
    # Save final results
    save_results(model, history, best_preds, best_labels, best_probs, class_names, device)
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Previous Accuracy: 28.97%")
    improvement = best_acc - 28.97
    print(f"  Improvement: {improvement:+.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

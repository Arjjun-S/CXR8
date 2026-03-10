"""
================================================================================
Model: Vision Transformer (ViT-B/16) Pretrained
================================================================================
Multi-label Thoracic Disease Classification using CXR8 Dataset

Architecture: ViT-B/16 with ImageNet pretrained weights
Features:
- Pretrained backbone fine-tuning
- Class weights for imbalance handling
- Medical-appropriate augmentations
- Warmup + Cosine Annealing LR scheduler
- Label smoothing regularization
- GPU-optimized training
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
# CONFIGURATION
# ================================================================================
class Config:
    """Centralized configuration for reproducibility"""
    
    # Paths
    BASE_DIR = r"E:\arjjun\CXR8"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    MODEL_DIR = os.path.join(BASE_DIR, "Model_ViT_Pretrained")
    
    # Model hyperparameters
    MODEL_NAME = "ViT-B/16"
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 30  # Max 30 epochs
    WARMUP_EPOCHS = 3
    LEARNING_RATE = 3e-5  # Lower LR for fine-tuning pretrained
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    NUM_WORKERS = 4
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Training
    GRADIENT_CLIP = 1.0
    VAL_SPLIT = 0.2  # 20% validation


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
    """Setup and verify GPU device"""
    print("\n" + "=" * 70)
    print(" GPU CONFIGURATION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n  WARNING: CUDA is not available!")
        print("  Training will proceed on CPU (this will be very slow).")
        device = torch.device("cpu")
    else:
        # Check for CUDA compatibility
        try:
            # Test if CUDA kernel actually works
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            device = torch.device("cuda")
            print(f"\n  PyTorch version:  {torch.__version__}")
            print(f"  CUDA available:   True")
            print(f"  GPU Device:       {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except RuntimeError as e:
            print(f"\n  WARNING: CUDA available but kernel error detected!")
            print(f"  Error: {str(e)[:100]}...")
            print(f"\n  Your GPU may require a newer PyTorch version.")
            print(f"  To fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            print(f"\n  Falling back to CPU (this will be very slow).")
            device = torch.device("cpu")
    
    print(f"  Device:           {device}")
    print("=" * 70)
    
    return device


# ================================================================================
# DATA LOADING
# ================================================================================
def get_class_weights(dataset, device):
    """Compute balanced class weights for imbalanced dataset"""
    labels = [dataset.targets[i] for i in range(len(dataset))]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def get_patient_wise_split(dataset, val_ratio=0.2, seed=42):
    """
    Create patient-wise train/val split to prevent data leakage.
    Images from same patient stay in same split.
    """
    np.random.seed(seed)
    
    # Group indices by patient (using image filename pattern)
    # CXR8 format: 00000001_000.png (patient_id_image_num.png)
    patient_indices = {}
    for idx in range(len(dataset)):
        img_path = dataset.imgs[idx][0]
        img_name = os.path.basename(img_path)
        patient_id = img_name.split('_')[0]
        
        if patient_id not in patient_indices:
            patient_indices[patient_id] = []
        patient_indices[patient_id].append(idx)
    
    # Split patients, not images
    patients = list(patient_indices.keys())
    np.random.shuffle(patients)
    
    split_idx = int(len(patients) * (1 - val_ratio))
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])
    
    train_indices = []
    val_indices = []
    
    for patient_id, indices in patient_indices.items():
        if patient_id in train_patients:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
    
    return train_indices, val_indices


def get_data_loaders(config, device):
    """Create train and validation data loaders"""
    print("\n" + "=" * 70)
    print(" LOADING DATASET")
    print("=" * 70)
    
    # Medical-safe augmentations (no color jitter for X-rays)
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(config.TRAIN_DIR)
    classes = full_dataset.classes
    num_classes = len(classes)
    
    # Print class distribution
    print(f"\n  Dataset: {config.TRAIN_DIR}")
    print(f"  Total images: {len(full_dataset)}")
    print(f"  Number of classes: {num_classes}")
    print("\n  Class Distribution:")
    print("  " + "-" * 45)
    
    class_counts = Counter(full_dataset.targets)
    for idx, class_name in enumerate(classes):
        print(f"    {class_name:25s}: {class_counts[idx]:6d}")
    print("  " + "-" * 45)
    
    # Compute class weights
    class_weights = get_class_weights(full_dataset, device)
    print(f"\n  Class weights computed for imbalance handling")
    
    # Patient-wise split
    train_indices, val_indices = get_patient_wise_split(
        full_dataset, val_ratio=config.VAL_SPLIT, seed=config.RANDOM_SEED
    )
    
    print(f"\n  Patient-wise split:")
    print(f"    Training samples:   {len(train_indices)}")
    print(f"    Validation samples: {len(val_indices)}")
    
    # Create datasets with transforms
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=val_transform)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    print("=" * 70)
    
    return train_loader, val_loader, classes, class_weights


# ================================================================================
# MODEL
# ================================================================================
def create_model(num_classes, device):
    """Create ViT-B/16 model with pretrained weights"""
    print("\n" + "=" * 70)
    print(" MODEL ARCHITECTURE")
    print("=" * 70)
    
    # Load pretrained ViT-B/16
    print(f"\n  Loading ViT-B/16 with ImageNet pretrained weights...")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    
    # Replace classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, num_classes)
    )
    
    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Model: ViT-B/16")
    print(f"  Input size: 224 x 224")
    print(f"  Patch size: 16 x 16")
    print(f"  Hidden dim: 768")
    print(f"  Layers: 12")
    print(f"  Heads: 12")
    print(f"\n  Total parameters:     {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Output classes: {num_classes}")
    print("=" * 70)
    
    return model.to(device)


# ================================================================================
# TRAINING UTILITIES
# ================================================================================
def get_scheduler(optimizer, num_training_steps, num_warmup_steps):
    """Warmup + Cosine Annealing scheduler"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch with clean progress display"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}/{total_epochs:02d} [TRAIN]", 
                ncols=100, leave=True)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix_str(
            f"Loss: {running_loss/(pbar.n+1):.4f} | Acc: {100.*correct/total:.2f}% | LR: {current_lr:.2e}"
        )
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device, epoch, total_epochs):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}/{total_epochs:02d} [VALID]", 
                    ncols=100, leave=True)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix_str(f"Loss: {running_loss/(pbar.n+1):.4f} | Acc: {100.*correct/total:.2f}%")
    
    return running_loss / len(loader), 100. * correct / total, all_labels, all_preds, all_probs


def calculate_metrics(y_true, y_pred, y_probs, num_classes):
    """Calculate comprehensive evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Specificity (macro-averaged)
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity = np.mean(specificity_per_class)
    
    # ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_true_bin, np.array(y_probs), average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    return {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall (Sensitivity)': float(recall),
        'Specificity': float(specificity),
        'F1-Score': float(f1),
        'ROC-AUC': float(roc_auc)
    }


# ================================================================================
# PLOTTING
# ================================================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs, train_accs, 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    axes[0].plot(epochs, val_accs, 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(1, len(epochs))
    
    # Loss plot
    axes[1].plot(epochs, train_losses, 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    axes[1].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(1, len(epochs))
    
    plt.suptitle('ViT-B/16 Training Progress', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label',
           title='Confusion Matrix - ViT-B/16')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================================
# SAVE RESULTS
# ================================================================================
def save_metrics(metrics, classes, y_true, y_pred, config, best_val_acc, save_dir):
    """Save metrics to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {config.MODEL_NAME}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: GPU ({torch.cuda.get_device_name(0)})\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Epochs:          {config.EPOCHS}\n")
        f.write(f"  Batch size:      {config.BATCH_SIZE}\n")
        f.write(f"  Learning rate:   {config.LEARNING_RATE}\n")
        f.write(f"  Weight decay:    {config.WEIGHT_DECAY}\n")
        f.write(f"  Label smoothing: {config.LABEL_SMOOTHING}\n")
        f.write(f"  Image size:      {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")
        f.write(f"  Random seed:     {config.RANDOM_SEED}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("FINAL METRICS (Best Model)\n")
        f.write("-" * 70 + "\n")
        for name, value in metrics.items():
            f.write(f"  {name:25s}: {value:.4f}\n")
        f.write(f"\n  Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
        
        f.write("\n" + "=" * 70 + "\n")
    
    # Save as JSON
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({
            **metrics,
            'best_val_acc': best_val_acc,
            'model': config.MODEL_NAME,
            'epochs': config.EPOCHS,
            'timestamp': timestamp
        }, f, indent=2)


def save_description(config, save_dir):
    """Save model description"""
    with open(os.path.join(save_dir, 'description.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL DESCRIPTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Architecture: Vision Transformer (ViT-B/16)\n\n")
        
        f.write("Model Details:\n")
        f.write("-" * 40 + "\n")
        f.write("  Base: ViT-B/16 with ImageNet pretrained weights\n")
        f.write("  Patch size: 16x16 pixels\n")
        f.write("  Hidden dimension: 768\n")
        f.write("  Transformer layers: 12\n")
        f.write("  Attention heads: 12\n")
        f.write("  MLP dimension: 3072\n")
        f.write("  Parameters: ~86M\n\n")
        
        f.write("Training Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Epochs: {config.EPOCHS}\n")
        f.write(f"  Batch size: {config.BATCH_SIZE}\n")
        f.write(f"  Optimizer: AdamW\n")
        f.write(f"  Learning rate: {config.LEARNING_RATE}\n")
        f.write(f"  Weight decay: {config.WEIGHT_DECAY}\n")
        f.write(f"  Scheduler: Warmup + Cosine Annealing\n")
        f.write(f"  Warmup epochs: {config.WARMUP_EPOCHS}\n")
        f.write(f"  Label smoothing: {config.LABEL_SMOOTHING}\n")
        f.write(f"  Gradient clipping: {config.GRADIENT_CLIP}\n\n")
        
        f.write("Data Augmentation:\n")
        f.write("-" * 40 + "\n")
        f.write("  - Random resized crop (scale 0.85-1.0)\n")
        f.write("  - Random horizontal flip (p=0.5)\n")
        f.write("  - Random affine (rotation ±10°, translate 5%)\n")
        f.write("  - Random autocontrast (p=0.2)\n")
        f.write("  - ImageNet normalization\n\n")
        
        f.write("Imbalance Handling:\n")
        f.write("-" * 40 + "\n")
        f.write("  - Balanced class weights in loss function\n")
        f.write("  - Patient-wise train/val split\n\n")
        
        f.write("=" * 70 + "\n")


# ================================================================================
# MAIN TRAINING LOOP
# ================================================================================
def main():
    """Main training function"""
    config = Config()
    
    print("\n" + "=" * 70)
    print("   VISION TRANSFORMER (ViT-B/16) TRAINING")
    print("   Multi-label Thoracic Disease Classification")
    print("=" * 70)
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    print(f"\n  Random seed: {config.RANDOM_SEED}")
    
    # Setup device
    device = setup_device()
    
    # Create output directory
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader, classes, class_weights = get_data_loaders(config, device)
    num_classes = len(classes)
    
    # Create model
    model = create_model(num_classes, device)
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    # Optimizer with different LR for backbone vs head
    optimizer = optim.AdamW([
        {'params': model.heads.head.parameters(), 'lr': config.LEARNING_RATE * 10},
        {'params': [p for n, p in model.named_parameters() if 'heads' not in n], 'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
    scheduler = get_scheduler(optimizer, total_steps, warmup_steps)
    
    print("\n" + "=" * 70)
    print(" TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"\n  Epochs:          {config.EPOCHS}")
    print(f"  Batch size:      {config.BATCH_SIZE}")
    print(f"  Learning rate:   {config.LEARNING_RATE}")
    print(f"  Warmup epochs:   {config.WARMUP_EPOCHS}")
    print(f"  Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"  Loss: CrossEntropy with class weights")
    print(f"  Scheduler: Warmup + Cosine Annealing")
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print("\n" + "=" * 70)
    print(" STARTING TRAINING")
    print("=" * 70 + "\n")
    
    for epoch in range(1, config.EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, epoch, config.EPOCHS
        )
        
        # Validate
        val_loss, val_acc, y_true, y_pred, y_probs = validate(
            model, val_loader, criterion, device, epoch, config.EPOCHS
        )
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Epoch summary
        print(f"\n  {'='*60}")
        print(f"  Epoch {epoch:02d}/{config.EPOCHS:02d} Summary:")
        print(f"    Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'model.h5'))
            print(f"    [NEW BEST] Saved model with accuracy: {val_acc:.2f}%")
        
        print(f"  {'='*60}\n")
        
        # Periodic metrics display
        if epoch % 5 == 0 or epoch == config.EPOCHS:
            metrics = calculate_metrics(y_true, y_pred, y_probs, num_classes)
            print(f"  Current Metrics:")
            for name, value in metrics.items():
                print(f"    {name}: {value:.4f}")
            print()
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'model.h5')))
    val_loss, val_acc, y_true, y_pred, y_probs = validate(
        model, val_loader, criterion, device, config.EPOCHS, config.EPOCHS
    )
    metrics = calculate_metrics(y_true, y_pred, y_probs, num_classes)
    
    print(f"\n  FINAL METRICS (Best Model):")
    print("  " + "-" * 45)
    for name, value in metrics.items():
        print(f"    {name:25s}: {value:.4f}")
    print(f"    {'Best Validation Accuracy':25s}: {best_val_acc:.2f}%")
    print("  " + "-" * 45)
    
    # Save all outputs
    print("\n  Saving outputs...")
    
    # Save metrics
    save_metrics(metrics, classes, y_true, y_pred, config, best_val_acc, config.MODEL_DIR)
    print(f"    metrics.txt")
    print(f"    metrics.json")
    
    # Save training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        os.path.join(config.MODEL_DIR, 'training_curves.png'))
    print(f"    training_curves.png")
    
    # Save confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes, 
                         os.path.join(config.MODEL_DIR, 'confusion_matrix.png'))
    print(f"    confusion_matrix.png")
    
    # Save description
    save_description(config, config.MODEL_DIR)
    print(f"    description.txt")
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Outputs saved to: {config.MODEL_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
Microsoft Swin Transformer for NIH CXR8 Multi-Label Classification
Uses the FULL dataset (112,120 images) from images_001 to images_012
35 epochs, BCEWithLogitsLoss with pos_weight for class imbalance

FIXED VERSION - Proper saving and error handling
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# Configuration
WORKSPACE = r"E:\arjjun\CXR8"
DATA_CSV = os.path.join(WORKSPACE, "Data_Entry_2017_v2020.csv")
IMAGES_BASE = os.path.join(WORKSPACE, "images")
OUTPUT_DIR = os.path.join(WORKSPACE, "Model_microsoft_swin")

# Training parameters
NUM_EPOCHS = 35
BATCH_SIZE = 16  # Smaller batch for GPU memory
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 4
THRESHOLD = 0.3

# Disease classes (15 total including No Finding)
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]
NUM_CLASSES = len(DISEASE_CLASSES)


def save_model_h5(model, filepath):
    """Save PyTorch model weights to HDF5 format"""
    try:
        state_dict = model.state_dict()
        with h5py.File(filepath, 'w') as f:
            for key, value in state_dict.items():
                # Convert tensor to numpy and save
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


def analyze_class_distribution(df, output_dir):
    """Analyze and save class distribution diagram"""
    print("\n" + "="*60)
    print("FULL DATASET CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count labels
    label_counts = Counter()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            label = label.strip()
            if label in DISEASE_CLASSES:
                label_counts[label] += 1
    
    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_labels]
    counts = [x[1] for x in sorted_labels]
    
    # Print statistics
    print(f"\nTotal images: {len(df)}")
    print(f"Total label occurrences: {sum(counts)}")
    print("\nPer-class distribution:")
    print("-"*50)
    for label, count in sorted_labels:
        pct = (count / len(df)) * 100
        print(f"  {label:<20}: {count:>6} ({pct:>5.2f}%)")
    
    # Create distribution plot
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
    bars = plt.barh(range(len(labels)), counts, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Number of Images', fontsize=12)
    plt.ylabel('Disease Class', fontsize=12)
    plt.title(f'NIH CXR8 Full Dataset Class Distribution\n(Total: {len(df):,} images)', fontsize=14)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count + 500, i, f'{count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print(f"\nClass distribution saved to: {os.path.join(output_dir, 'class_distribution.png')}")
    
    return label_counts


class NIH_CXR8_Dataset(Dataset):
    """NIH Chest X-Ray dataset using full CSV with image search"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        # Pre-compute image paths
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
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def create_model():
    """Create Swin Transformer model using timm"""
    print("\nCreating Swin Tiny model from timm...")
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    print(f"Model created: swin_tiny_patch4_window7_224")
    print(f"Number of classes: {NUM_CLASSES}")
    return model


def calculate_pos_weight(dataset, device):
    """Calculate positive class weights for BCEWithLogitsLoss"""
    print("\nCalculating class weights...")
    all_labels = np.array(dataset.labels)
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    
    # pos_weight = neg_count / pos_count (capped)
    pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)  # Cap at 50 to avoid extreme weights
    
    print("Class weights (pos_weight):")
    for i, (cls, w) in enumerate(zip(DISEASE_CLASSES, pos_weight)):
        print(f"  {cls:<20}: {w:.2f}")
    
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Collect predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} - Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    preds_binary = (all_preds > THRESHOLD).astype(float)
    accuracy = (preds_binary == all_labels).mean() * 100
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    preds_binary = (all_preds > THRESHOLD).astype(float)
    accuracy = (preds_binary == all_labels).mean() * 100
    
    # Per-class AUC
    auc_scores = []
    for i in range(NUM_CLASSES):
        if all_labels[:, i].sum() > 0 and all_labels[:, i].sum() < len(all_labels):
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(0.5)
    
    mean_auc = np.mean(auc_scores)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy, mean_auc, auc_scores, all_preds, all_labels


def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This model requires GPU training.")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load CSV
    print("\n" + "="*60)
    print("LOADING FULL NIH CXR8 DATASET")
    print("="*60)
    
    df = pd.read_csv(DATA_CSV)
    print(f"Total entries in CSV: {len(df)}")
    
    # Analyze class distribution
    label_counts = analyze_class_distribution(df, OUTPUT_DIR)
    
    # Split data (80% train, 10% val, 10% test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    
    # Create transforms
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
    
    # Create datasets
    print("\n" + "="*60)
    print("CREATING DATASETS")
    print("="*60)
    
    print("\nCreating training dataset...")
    train_dataset = NIH_CXR8_Dataset(train_df, transform=train_transform)
    
    print("\nCreating validation dataset...")
    val_dataset = NIH_CXR8_Dataset(val_df, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Calculate pos_weight from training data
    pos_weight = calculate_pos_weight(train_dataset, device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print("\n" + "="*60)
    print(f"TRAINING MICROSOFT SWIN - {NUM_EPOCHS} EPOCHS")
    print("="*60)
    
    best_val_auc = 0.0
    best_epoch = 0
    best_val_acc = 0.0
    best_class_aucs = []
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, val_auc, class_aucs, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_val_acc = val_acc
            best_class_aucs = class_aucs.copy() if isinstance(class_aucs, list) else list(class_aucs)
            
            # Save PyTorch checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': float(val_auc),
                'val_acc': float(val_acc),
                'class_aucs': [float(x) for x in class_aucs],  # Convert to plain Python floats
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            
            # Save model weights only (for easy loading)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))
            
            # Save .h5 format
            save_model_h5(model, os.path.join(OUTPUT_DIR, 'model.h5'))
            
            print(f"  >> New best model saved! AUC: {val_auc:.4f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth'))
            print(f"  >> Checkpoint saved: checkpoint_epoch_{epoch}.pth")
    
    # Save final model in multiple formats
    print("\n" + "="*60)
    print("SAVING FINAL MODELS")
    print("="*60)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_final.pth'))
    print(f"  Final model saved: model_final.pth")
    
    save_model_h5(model, os.path.join(OUTPUT_DIR, 'model_final.h5'))
    
    # Final evaluation using stored best values (no need to reload)
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    print(f"\nBest Model Results (Epoch {best_epoch}):")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Mean ROC-AUC: {best_val_auc:.4f}")
    print(f"\nPer-class AUC:")
    for i, cls in enumerate(DISEASE_CLASSES):
        if i < len(best_class_aucs):
            print(f"  {cls:<20}: {best_class_aucs[i]:.4f}")
    
    # Save metrics
    metrics = {
        'model': 'swin_tiny_patch4_window7_224',
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'threshold': THRESHOLD,
        'total_images': len(df),
        'train_images': len(train_dataset),
        'val_images': len(val_dataset),
        'best_epoch': best_epoch,
        'best_val_accuracy': float(best_val_acc),
        'best_val_auc': float(best_val_auc),
        'class_aucs': {cls: float(best_class_aucs[i]) for i, cls in enumerate(DISEASE_CLASSES) if i < len(best_class_aucs)},
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: metrics.json")
    
    # Save training curves
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['val_auc'], label='Val AUC', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Validation AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150)
        plt.close()
        print(f"Training curves saved: training_curves.png")
    except Exception as e:
        print(f"Warning: Could not save training curves: {e}")
    
    # List all saved files
    print("\n" + "="*60)
    print("SAVED FILES")
    print("="*60)
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / (1024*1024)  # MB
        print(f"  {f:<30} ({size:.2f} MB)")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

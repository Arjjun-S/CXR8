"""Quick final evaluation for Microsoft Swin model"""
import os
import json
import numpy as np
import torch
import timm

WORKSPACE = r"E:\arjjun\CXR8"
OUTPUT_DIR = os.path.join(WORKSPACE, "Model_microsoft_swin")

DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

# Load checkpoint
checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), weights_only=False)

print("="*60)
print("MICROSOFT SWIN - FINAL RESULTS")
print("="*60)
print(f"\nBest model from epoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
print(f"Mean ROC-AUC: {checkpoint['val_auc']:.4f}")

print("\nPer-class AUC:")
for i, cls in enumerate(DISEASE_CLASSES):
    print(f"  {cls:<20}: {checkpoint['class_aucs'][i]:.4f}")

# Save to metrics file
metrics = {
    'model': 'swin_tiny_patch4_window7_224',
    'best_epoch': checkpoint['epoch'],
    'best_val_accuracy': checkpoint['val_acc'],
    'best_val_auc': checkpoint['val_auc'],
    'class_aucs': {cls: float(auc) for cls, auc in zip(DISEASE_CLASSES, checkpoint['class_aucs'])}
}

with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: {os.path.join(OUTPUT_DIR, 'final_metrics.json')}")
print("\nTraining complete!")

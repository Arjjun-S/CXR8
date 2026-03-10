"""
================================================================================
CXR8 Dataset Analysis and Preprocessing
================================================================================
This script performs comprehensive data analysis on the CXR8 dataset:
- Loads dataset metadata
- Computes class counts and distribution
- Calculates imbalance ratio
- Generates visualizations
- Saves analysis reports

Run this BEFORE training any model to understand your data.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# ===================== CONFIGURATION =====================
BASE_DIR = r"E:\arjjun\CXR8"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "Preprocessing")
CSV_FILE = os.path.join(BASE_DIR, "Data_Entry_2017_v2020.csv")

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Disease classes
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No_Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


def count_images_in_train_folder():
    """Count images in each class folder"""
    print("\n" + "=" * 60)
    print("COUNTING IMAGES IN TRAIN FOLDER")
    print("=" * 60)
    
    class_counts = {}
    total_images = 0
    
    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
            class_counts[cls] = count
            total_images += count
            print(f"  {cls:25s}: {count:6d} images")
        else:
            class_counts[cls] = 0
            print(f"  {cls:25s}: NOT FOUND")
    
    print("-" * 60)
    print(f"  {'TOTAL':25s}: {total_images:6d} images")
    
    return class_counts, total_images


def calculate_imbalance_metrics(class_counts):
    """Calculate various imbalance metrics"""
    print("\n" + "=" * 60)
    print("IMBALANCE ANALYSIS")
    print("=" * 60)
    
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Imbalance ratio (max/min)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Coefficient of variation
    cv = std_count / mean_count if mean_count > 0 else 0
    
    # Find majority and minority classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    majority_class = sorted_classes[0]
    minority_class = sorted_classes[-1]
    
    metrics = {
        "max_count": int(max_count),
        "min_count": int(min_count),
        "mean_count": float(mean_count),
        "std_count": float(std_count),
        "imbalance_ratio": float(imbalance_ratio),
        "coefficient_of_variation": float(cv),
        "majority_class": majority_class[0],
        "majority_count": int(majority_class[1]),
        "minority_class": minority_class[0],
        "minority_count": int(minority_class[1]),
    }
    
    print(f"\n  Maximum class count:     {max_count}")
    print(f"  Minimum class count:     {min_count}")
    print(f"  Mean class count:        {mean_count:.2f}")
    print(f"  Std deviation:           {std_count:.2f}")
    print(f"\n  Imbalance Ratio:         {imbalance_ratio:.2f}x")
    print(f"  Coefficient of Variation: {cv:.2f}")
    print(f"\n  Majority class: {majority_class[0]} ({majority_class[1]} images)")
    print(f"  Minority class: {minority_class[0]} ({minority_class[1]} images)")
    
    return metrics


def calculate_class_weights(class_counts):
    """Calculate class weights for handling imbalance during training"""
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    weights = {}
    for cls, count in class_counts.items():
        if count > 0:
            # Balanced class weight formula: n_samples / (n_classes * n_samples_class)
            weights[cls] = total / (n_classes * count)
        else:
            weights[cls] = 1.0
    
    return weights


def plot_class_distribution(class_counts, output_path):
    """Generate bar plot of class distribution"""
    print("\n  Generating class distribution plot...")
    
    # Sort by count for better visualization
    sorted_data = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color gradient based on count
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(classes)))[::-1]
    
    bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('CXR8 Dataset - Class Distribution\n(Images per Class in Training Set)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add mean line
    mean_count = np.mean(counts)
    ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.0f}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def save_imbalance_report(class_counts, imbalance_metrics, class_weights, output_path):
    """Save detailed imbalance report to text file"""
    print("\n  Generating imbalance report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CXR8 DATASET - CLASS IMBALANCE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Dataset: CXR8 (NIH Chest X-ray)\n")
        f.write(f"Train folder: {TRAIN_DIR}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 70 + "\n\n")
        
        sorted_data = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(class_counts.values())
        
        f.write(f"{'Class':<25} {'Count':>10} {'Percentage':>12} {'Weight':>10}\n")
        f.write("-" * 57 + "\n")
        
        for cls, count in sorted_data:
            pct = (count / total * 100) if total > 0 else 0
            weight = class_weights.get(cls, 1.0)
            f.write(f"{cls:<25} {count:>10,} {pct:>11.2f}% {weight:>10.4f}\n")
        
        f.write("-" * 57 + "\n")
        f.write(f"{'TOTAL':<25} {total:>10,} {100:>11.2f}%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("IMBALANCE METRICS\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(f"{'Metric':<35} {'Value':>20}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Maximum class count':<35} {imbalance_metrics['max_count']:>20,}\n")
        f.write(f"{'Minimum class count':<35} {imbalance_metrics['min_count']:>20,}\n")
        f.write(f"{'Mean class count':<35} {imbalance_metrics['mean_count']:>20,.2f}\n")
        f.write(f"{'Standard deviation':<35} {imbalance_metrics['std_count']:>20,.2f}\n")
        f.write(f"{'Imbalance ratio (max/min)':<35} {imbalance_metrics['imbalance_ratio']:>20.2f}x\n")
        f.write(f"{'Coefficient of variation':<35} {imbalance_metrics['coefficient_of_variation']:>20.4f}\n\n")
        
        f.write(f"Majority class: {imbalance_metrics['majority_class']} ({imbalance_metrics['majority_count']:,} images)\n")
        f.write(f"Minority class: {imbalance_metrics['minority_class']} ({imbalance_metrics['minority_count']:,} images)\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n\n")
        
        if imbalance_metrics['imbalance_ratio'] > 10:
            f.write("HIGH IMBALANCE DETECTED!\n")
            f.write("Recommendations:\n")
            f.write("  1. Use class weights during training (computed above)\n")
            f.write("  2. Consider oversampling minority classes\n")
            f.write("  3. Use focal loss or other imbalance-aware losses\n")
            f.write("  4. Report per-class metrics, not just accuracy\n")
        elif imbalance_metrics['imbalance_ratio'] > 3:
            f.write("MODERATE IMBALANCE DETECTED\n")
            f.write("Recommendations:\n")
            f.write("  1. Use class weights during training\n")
            f.write("  2. Monitor per-class performance during training\n")
        else:
            f.write("Dataset is relatively balanced.\n")
            f.write("Standard training approaches should work well.\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"  Saved: {output_path}")


def main():
    """Main analysis function"""
    print("=" * 70)
    print("CXR8 DATASET ANALYSIS")
    print("=" * 70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Count images
    class_counts, total_images = count_images_in_train_folder()
    
    if total_images == 0:
        print("\nERROR: No images found in train folder!")
        print("Please run prepare_data.py first to copy images to train folder.")
        sys.exit(1)
    
    # Calculate imbalance metrics
    imbalance_metrics = calculate_imbalance_metrics(class_counts)
    
    # Calculate class weights
    class_weights = calculate_class_weights(class_counts)
    
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)
    
    # Generate plots
    plot_class_distribution(
        class_counts, 
        os.path.join(OUTPUT_DIR, "class_distribution.png")
    )
    
    # Save imbalance report
    save_imbalance_report(
        class_counts, 
        imbalance_metrics, 
        class_weights,
        os.path.join(OUTPUT_DIR, "imbalance_report.txt")
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - class_distribution.png")
    print("  - imbalance_report.txt")
    print("\nYou can now proceed with model training!")
    print("=" * 70)


if __name__ == "__main__":
    main()

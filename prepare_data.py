"""
Data Preparation Script for CXR8 Dataset
Copies 200 images per classification into train/classification folders
"""

import os
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Configuration
BASE_DIR = r"E:\arjjun\CXR8"
CSV_FILE = os.path.join(BASE_DIR, "Data_Entry_2017_v2020.csv")
IMAGE_DIRS = [os.path.join(BASE_DIR, "images", f"images_{str(i).zfill(3)}", "images") 
              for i in range(1, 13)]
OUTPUT_DIR = os.path.join(BASE_DIR, "train")
IMAGES_PER_CLASS = 5000

# All possible classifications in CXR8
CLASSIFICATIONS = [
    "Atelectasis",
    "Cardiomegaly", 
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

def find_image(image_name):
    """Find the full path of an image across all image directories"""
    for img_dir in IMAGE_DIRS:
        img_path = os.path.join(img_dir, image_name)
        if os.path.exists(img_path):
            return img_path
    return None

def main():
    print("Loading CSV data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Total images in dataset: {len(df)}")
    
    # Create output directories
    for cls in CLASSIFICATIONS:
        cls_dir = os.path.join(OUTPUT_DIR, cls.replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)
    
    # Track how many images we've copied per class
    class_counts = defaultdict(int)
    
    # Process images - only take single-label images for cleaner classification
    print("\nProcessing images...")
    
    for classification in CLASSIFICATIONS:
        print(f"\nProcessing {classification}...")
        class_dir = os.path.join(OUTPUT_DIR, classification.replace(" ", "_"))
        
        # Filter for images with only this single label
        if classification == "No Finding":
            class_df = df[df["Finding Labels"] == "No Finding"]
        else:
            # Get images where this is the ONLY label (no multi-label)
            class_df = df[df["Finding Labels"] == classification]
        
        print(f"  Found {len(class_df)} single-label images for {classification}")
        
        # If not enough single-label, also include multi-label where this is primary
        if len(class_df) < IMAGES_PER_CLASS:
            print(f"  Also including multi-label images...")
            multi_df = df[df["Finding Labels"].str.contains(classification, na=False)]
            # Exclude already included single-label ones
            multi_df = multi_df[~multi_df.index.isin(class_df.index)]
            class_df = pd.concat([class_df, multi_df])
        
        # Take up to IMAGES_PER_CLASS
        class_df = class_df.head(IMAGES_PER_CLASS)
        
        copied = 0
        for _, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"  Copying {classification}"):
            image_name = row["Image Index"]
            src_path = find_image(image_name)
            
            if src_path:
                dst_path = os.path.join(class_dir, image_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    copied += 1
        
        class_counts[classification] = copied
        print(f"  Copied {copied} images for {classification}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY - Images copied per class:")
    print("="*50)
    total = 0
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
        total += count
    print(f"\nTotal images copied: {total}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

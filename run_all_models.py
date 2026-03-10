"""
================================================================================
Sequential Model Runner
================================================================================
Runs all models sequentially: Swin → DenseNet121 → DenseNet2
Each model runs for 10 epochs on GPU (RTX)
================================================================================
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Configuration
BASE_DIR = r"E:\arjjun\CXR8"
PYTHON_PATH = r"E:\arjjun\CXR8\.venv312\Scripts\python.exe"

# Models to run in order
MODELS = [
    {
        "name": "Swin-T",
        "dir": "Model_Swin",
        "script": "train.py"
    },
    {
        "name": "DenseNet121",
        "dir": "Model_DenseNet121",
        "script": "train.py"
    },
    {
        "name": "DenseNet2-Optimized",
        "dir": "Model_DenseNet2",
        "script": "train.py"
    }
]


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def run_model(model_info):
    """Run a single model training"""
    name = model_info["name"]
    model_dir = os.path.join(BASE_DIR, model_info["dir"])
    script_path = os.path.join(model_dir, model_info["script"])
    
    print_header(f"STARTING: {name}")
    print(f"  Directory: {model_dir}")
    print(f"  Script: {script_path}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [PYTHON_PATH, script_path],
            cwd=model_dir,
            check=True
        )
        
        elapsed = time.time() - start_time
        print_header(f"COMPLETED: {name}")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if metrics.txt was created
        metrics_path = os.path.join(model_dir, "metrics.txt")
        if os.path.exists(metrics_path):
            print(f"  Metrics saved: {metrics_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {str(e)}")
        return False


def main():
    """Run all models sequentially"""
    print("\n")
    print("=" * 70)
    print("   SEQUENTIAL MODEL TRAINER")
    print("   CXR8 Multi-label Disease Classification")
    print("=" * 70)
    print(f"\n  Python: {PYTHON_PATH}")
    print(f"  Models to train: {len(MODELS)}")
    print(f"  Order: {' → '.join([m['name'] for m in MODELS])}")
    print(f"\n  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    results = []
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n\n{'#' * 70}")
        print(f"  MODEL {i}/{len(MODELS)}: {model['name']}")
        print(f"{'#' * 70}")
        
        success = run_model(model)
        results.append({
            "name": model["name"],
            "success": success
        })
        
        if not success:
            print(f"\n  WARNING: {model['name']} failed. Continuing to next model...")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n\n")
    print("=" * 70)
    print("   TRAINING SUMMARY")
    print("=" * 70)
    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")
    print(f"\n  Results:")
    for r in results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        print(f"    {r['name']:25s}: {status}")
    
    print("\n  Metrics saved to:")
    for model in MODELS:
        metrics_path = os.path.join(BASE_DIR, model["dir"], "metrics.txt")
        if os.path.exists(metrics_path):
            print(f"    - {metrics_path}")
    
    print("\n" + "=" * 70)
    print("   ALL TRAINING COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

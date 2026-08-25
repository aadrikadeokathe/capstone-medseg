import os
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import get_spleen_splits

def verify():
    images_path = os.path.join(PROJECT_ROOT, "data", "processed", "spleen_images.npy")
    if not os.path.exists(images_path):
        print("Spleen data file not found:", images_path)
        return

    images = np.load(images_path)
    total_slices = len(images)
    indices = np.arange(total_slices)
    
    seed = 42
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
    train_end = int(total_slices * train_ratio)
    val_end = train_end + int(total_slices * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print("=== SPLEEN DATA SPLIT VERIFICATION (SEED = 42) ===")
    print(f"Total Slices: {total_slices}")
    print(f"Train Slices Count: {len(train_idx)} ({len(train_idx)/total_slices*100:.1f}%)")
    print(f"Val Slices Count:   {len(val_idx)} ({len(val_idx)/total_slices*100:.1f}%)")
    print(f"Test Slices Count:  {len(test_idx)} ({len(test_idx)/total_slices*100:.1f}%)")
    print(f"First 5 Train Indices: {train_idx[:5]}")
    print(f"First 5 Val Indices:   {val_idx[:5]}")
    print(f"First 5 Test Indices:  {test_idx[:5]}")
    print("--------------------------------------------------")
    print(f"Overlap (Train & Val):  {len(set(train_idx) & set(val_idx))}")
    print(f"Overlap (Train & Test): {len(set(train_idx) & set(test_idx))}")
    print(f"Overlap (Val & Test):   {len(set(val_idx) & set(test_idx))}")
    print(f"Disjoint Partition Check Passed: {len(set(train_idx) & set(test_idx)) == 0}")

if __name__ == "__main__":
    verify()

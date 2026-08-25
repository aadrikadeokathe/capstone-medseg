import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import get_spleen_splits

def check_seed_consistency():
    tr1, val1, te1, _ = get_spleen_splits(seed=42)
    tr2, val2, te2, _ = get_spleen_splits(seed=42)

    # Get underlying raw indices for seed 42
    images_path = os.path.join(PROJECT_ROOT, "data", "processed", "spleen_images.npy")
    images = np.load(images_path)
    total_slices = len(images)
    
    indices1 = np.arange(total_slices)
    rng1 = np.random.RandomState(42)
    rng1.shuffle(indices1)

    train_end = int(total_slices * 0.70)
    val_end = train_end + int(total_slices * 0.15)

    train_idx = indices1[:train_end]
    val_idx = indices1[train_end:val_end]
    test_idx = indices1[val_end:]

    # Run again
    indices2 = np.arange(total_slices)
    rng2 = np.random.RandomState(42)
    rng2.shuffle(indices2)

    train_idx2 = indices2[:train_end]
    val_idx2 = indices2[train_end:val_end]
    test_idx2 = indices2[val_end:]

    print("=== SEED CONSISTENCY VERIFICATION ===")
    print("Train indices identical:", np.array_equal(train_idx, train_idx2))
    print("Val indices identical:  ", np.array_equal(val_idx, val_idx2))
    print("Test indices identical: ", np.array_equal(test_idx, test_idx2))
    
    print("\nExact Slice Index Lists (Seed = 42):")
    print(f"  Total Slices: {total_slices}")
    print(f"  Train Indices ({len(train_idx)}): {train_idx.tolist()}")
    print(f"  Val Indices   ({len(val_idx)}): {val_idx.tolist()}")
    print(f"  Test Indices  ({len(test_idx)}): {test_idx.tolist()}")

if __name__ == "__main__":
    check_seed_consistency()

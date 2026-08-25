import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import build_liver_dataset

if __name__ == "__main__":
    print("Building liver dataset...")
    images, masks = build_liver_dataset()
    print("Liver dataset built successfully!")
    print("  Images shape:", images.shape)
    print("  Masks shape: ", masks.shape)

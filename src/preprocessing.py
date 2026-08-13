# preprocessing.py
"""
Preprocessing utilities for 3D CT spleen volumes.

Extracts 2D slices from NIfTI volumes, filters for slices containing
the target organ, normalises intensity, and saves processed arrays.
"""

import os
import numpy as np
import nibabel as nib

# Resolve project root from this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def extract_slices(scan_path, mask_path, min_mask_pixels=50):
    """
    Load a 3D CT scan and its mask, and return 2D slices that contain
    enough foreground (spleen) pixels.

    Parameters
    ----------
    scan_path : str
        Path to the CT scan NIfTI file (.nii.gz).
    mask_path : str
        Path to the corresponding label NIfTI file (.nii.gz).
    min_mask_pixels : int, optional
        Minimum number of nonzero mask pixels required to keep a slice.
        Default is 50.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (image_slice, mask_slice) tuples where:
        - image_slice is float32, intensity normalised to [0, 1]
        - mask_slice  is uint8,  binary {0, 1}
    """
    # Load volumes
    scan_vol = nib.load(scan_path).get_fdata()   # (H, W, D)
    mask_vol = nib.load(mask_path).get_fdata()    # (H, W, D)

    slices = []
    n_slices = scan_vol.shape[2]

    for i in range(n_slices):
        mask_2d = mask_vol[:, :, i]

        # Skip slices with insufficient foreground
        if np.count_nonzero(mask_2d) < min_mask_pixels:
            continue

        img_2d = scan_vol[:, :, i].astype(np.float32)

        # Min-max normalise CT intensities to [0, 1]
        img_min, img_max = img_2d.min(), img_2d.max()
        if img_max - img_min > 0:
            img_2d = (img_2d - img_min) / (img_max - img_min)
        else:
            img_2d = np.zeros_like(img_2d)

        mask_2d = mask_2d.astype(np.uint8)

        slices.append((img_2d, mask_2d))

    return slices


def build_spleen_dataset(output_dir=None):
    """
    Process all training scan/label pairs and save as two numpy arrays:
      - spleen_images.npy  (N, H, W) float32
      - spleen_masks.npy   (N, H, W) uint8

    Parameters
    ----------
    output_dir : str, optional
        Directory to write the .npy files.  Defaults to data/processed/.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (images, masks) arrays.
    """
    # Import here to avoid circular imports
    from src.data_loading import list_spleen_files

    if output_dir is None:
        output_dir = PROCESSED_DIR
    os.makedirs(output_dir, exist_ok=True)

    pairs = list_spleen_files()
    all_images = []
    all_masks = []

    for idx, (scan_path, mask_path) in enumerate(pairs):
        basename = os.path.basename(scan_path)
        slices = extract_slices(scan_path, mask_path)
        print(f"  [{idx+1:02d}/{len(pairs)}] {basename}: {len(slices)} slices extracted")
        for img, msk in slices:
            all_images.append(img)
            all_masks.append(msk)

    images = np.stack(all_images, axis=0)  # (N, H, W)
    masks = np.stack(all_masks, axis=0)    # (N, H, W)

    img_path = os.path.join(output_dir, "spleen_images.npy")
    msk_path = os.path.join(output_dir, "spleen_masks.npy")
    np.save(img_path, images)
    np.save(msk_path, masks)

    print(f"\n[INFO] Saved {images.shape[0]} slices")
    print(f"  Images: {img_path}  shape={images.shape}  dtype={images.dtype}")
    print(f"  Masks:  {msk_path}  shape={masks.shape}  dtype={masks.dtype}")

    return images, masks


if __name__ == "__main__":
    build_spleen_dataset()

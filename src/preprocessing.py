# preprocessing.py
"""
Preprocessing utilities for 3D CT spleen and 4D MRI brain tumour volumes.
"""

import os
import gc
import numpy as np
import nibabel as nib
from src.data_loading import (
    list_spleen_files,
    list_braintumour_files,
    download_braintumour_dataset,
)

# Resolve project root from this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ==========================================
# Spleen Preprocessing
# ==========================================

def extract_slices(scan_path, mask_path, min_mask_pixels=50):
    """
    Load a 3D CT scan and its mask, and return 2D slices that contain
    enough foreground (spleen) pixels.
    """
    scan_vol = nib.load(scan_path).get_fdata()   # (H, W, D)
    mask_vol = nib.load(mask_path).get_fdata()    # (H, W, D)

    slices = []
    n_slices = scan_vol.shape[2]

    for i in range(n_slices):
        mask_2d = mask_vol[:, :, i]

        if np.count_nonzero(mask_2d) < min_mask_pixels:
            continue

        img_2d = scan_vol[:, :, i].astype(np.float32)

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
    """
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


# ==========================================
# Brain Tumour Preprocessing
# ==========================================

def extract_slices_braintumour(scan_path: str, mask_path: str, min_mask_pixels: int = 50):
    """
    Loads 4D scan (X, Y, slices, 4 channels) and 3D mask using nibabel in float32.
    Filters slices with > min_mask_pixels nonzero pixels across ANY tumor class (1, 2, 3).
    Normalizes each of the 4 channels independently to [0, 1] range per slice.
    Converts multi-class mask into a binary 'tumor present or not' mask (merge classes 1,2,3 into 1).
    
    Returns:
        List of (image_slice [shape H, W, 4], binary_mask_slice [shape H, W]) tuples.
    """
    scan_nii = nib.load(scan_path)
    mask_nii = nib.load(mask_path)
    
    scan_data = scan_nii.get_fdata(dtype=np.float32)  # shape (H, W, depth, 4)
    mask_data = mask_nii.get_fdata(dtype=np.float32)  # shape (H, W, depth)

    num_slices = scan_data.shape[2]
    extracted_slices = []

    for s in range(num_slices):
        mask_slice = mask_data[:, :, s]
        
        if np.count_nonzero(mask_slice) > min_mask_pixels:
            img_slice = scan_data[:, :, s, :]  # shape (H, W, 4)
            
            norm_img_slice = np.zeros_like(img_slice, dtype=np.float32)
            for c in range(4):
                chan = img_slice[:, :, c]
                c_min, c_max = chan.min(), chan.max()
                if c_max > c_min:
                    norm_img_slice[:, :, c] = (chan - c_min) / (c_max - c_min)
                else:
                    norm_img_slice[:, :, c] = 0.0

            binary_mask_slice = (mask_slice > 0).astype(np.float32)

            extracted_slices.append((norm_img_slice, binary_mask_slice))

    del scan_data, mask_data, scan_nii, mask_nii
    gc.collect()

    return extracted_slices


def build_braintumour_dataset(data_dir: str = "data", output_dir: str = "data/processed", min_mask_pixels: int = 50):
    """
    Loops through all scan/label pairs, extracts slices, and saves them as
    data/processed/braintumour_images.npy and data/processed/braintumour_masks.npy
    using memory-mapped arrays to prevent RAM overflow.
    Prints final shapes and total slice count.
    """
    pairs = list_braintumour_files(data_dir=data_dir)
    
    print("Pass 1/2: Counting valid slices across all volumes...")
    slice_counts = []
    for idx, (_, mask_path) in enumerate(pairs):
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata(dtype=np.float32)
        count = 0
        for s in range(mask_data.shape[2]):
            if np.count_nonzero(mask_data[:, :, s]) > min_mask_pixels:
                count += 1
        slice_counts.append(count)
        del mask_data, mask_nii
        if (idx + 1) % 100 == 0 or (idx + 1) == len(pairs):
            gc.collect()

    total_slices = sum(slice_counts)
    print(f"Total valid slices found across {len(pairs)} volumes: {total_slices}")

    if total_slices == 0:
        raise ValueError("No valid slices found with min_mask_pixels condition.")

    first_slices = extract_slices_braintumour(pairs[0][0], pairs[0][1], min_mask_pixels=min_mask_pixels)
    sample_img, sample_mask = first_slices[0]
    H, W, C = sample_img.shape

    os.makedirs(output_dir, exist_ok=True)
    images_path = os.path.join(output_dir, "braintumour_images.npy")
    masks_path = os.path.join(output_dir, "braintumour_masks.npy")

    print(f"Pass 2/2: Writing {total_slices} slices to memory-mapped npy files...")
    images_mm = np.lib.format.open_memmap(images_path, mode='w+', dtype=np.float32, shape=(total_slices, H, W, C))
    masks_mm = np.lib.format.open_memmap(masks_path, mode='w+', dtype=np.float32, shape=(total_slices, H, W))

    curr_idx = 0
    for idx, (scan_path, mask_path) in enumerate(pairs):
        n_slices = slice_counts[idx]
        if n_slices == 0:
            continue
        slices = extract_slices_braintumour(scan_path, mask_path, min_mask_pixels=min_mask_pixels)
        for img_s, mask_s in slices:
            images_mm[curr_idx] = img_s
            masks_mm[curr_idx] = mask_s
            curr_idx += 1
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(pairs):
            images_mm.flush()
            masks_mm.flush()
            print(f"Written {curr_idx}/{total_slices} slices ({idx + 1}/{len(pairs)} volumes).")
            gc.collect()

    images_mm.flush()
    masks_mm.flush()

    print(f"Successfully saved processed dataset to {output_dir}:")
    print(f"  Images shape: {images_mm.shape}")
    print(f"  Masks shape:  {masks_mm.shape}")
    print(f"  Total extracted slice count: {total_slices}")

    return images_mm, masks_mm


if __name__ == "__main__":
    build_spleen_dataset()

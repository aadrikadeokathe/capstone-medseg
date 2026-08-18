import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class InContextDataset(Dataset):
    """
    In-Context Learning Dataset for Few-Shot / Zero-Shot Segmentation.
    
    Each sample returns:
      - query_img: [C, H, W] tensor (target image to segment)
      - query_mask: [1, H, W] tensor (ground-truth mask for loss computation)
      - support_imgs: [S, C, H, W] tensor (support set context images)
      - support_masks: [S, 1, H, W] tensor (support set context binary masks)
    """

    def __init__(self, images: np.ndarray, masks: np.ndarray, num_support: int = 2, target_size: tuple = (128, 128), seed: int = 42):
        self.images = images
        self.masks = masks
        self.num_support = num_support
        self.target_size = target_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.images)

    def _resize(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        if array.ndim == 2:
            # Single channel image (H, W) -> [1, 1, H, W]
            tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif array.ndim == 3:
            # Multi-channel image (H, W, C) -> [1, C, H, W]
            tensor = torch.tensor(array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected array dimension: {array.ndim}")

        mode = "nearest" if is_mask else "bilinear"
        align_corners = None if is_mask else False
        resized = torch.nn.functional.interpolate(tensor, size=self.target_size, mode=mode, align_corners=align_corners)
        return resized.squeeze(0)  # [C, H, W] or [1, H, W]

    def __getitem__(self, idx: int):
        # Query sample
        query_img = self._resize(self.images[idx], is_mask=False)
        query_mask = self._resize(self.masks[idx] > 0, is_mask=True)

        # Support set selection (exclude current query slice)
        available_indices = [i for i in range(len(self.images)) if i != idx]
        if len(available_indices) < self.num_support:
            support_indices = available_indices
        else:
            support_indices = self.rng.choice(available_indices, size=self.num_support, replace=False)

        support_imgs_list = [self._resize(self.images[si], is_mask=False) for si in support_indices]
        support_masks_list = [self._resize(self.masks[si] > 0, is_mask=True) for si in support_indices]

        support_imgs = torch.stack(support_imgs_list, dim=0)   # [S, C, H, W]
        support_masks = torch.stack(support_masks_list, dim=0) # [S, 1, H, W]

        return query_img, query_mask, support_imgs, support_masks


def get_spleen_splits(
    data_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_support: int = 2,
    target_size: tuple = (128, 128)
):
    """
    Loads spleen images and masks, splits into train/val/test sets, and creates Datasets.
    """
    images_path = os.path.join(data_dir, "spleen_images.npy")
    masks_path = os.path.join(data_dir, "spleen_masks.npy")

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        raise FileNotFoundError(f"Processed spleen data not found at {images_path} and {masks_path}")

    images = np.load(images_path)  # (N, H, W)
    masks = np.load(masks_path)    # (N, H, W)

    total_slices = len(images)
    indices = np.arange(total_slices)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_end = int(total_slices * train_ratio)
    val_end = train_end + int(total_slices * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print("==================================================")
    print(" Data Split Statistics (MSD Task09 Spleen)")
    print("==================================================")
    print(f"  Total Slices: {total_slices}")
    print(f"  Train Set:    {len(train_idx)} slices ({len(train_idx)/total_slices*100:.1f}%)")
    print(f"  Val Set:      {len(val_idx)} slices ({len(val_idx)/total_slices*100:.1f}%)")
    print(f"  Test Set:     {len(test_idx)} slices ({len(test_idx)/total_slices*100:.1f}%)")
    print("--------------------------------------------------")

    train_dataset = InContextDataset(images[train_idx], masks[train_idx], num_support=num_support, target_size=target_size, seed=seed)
    val_dataset = InContextDataset(images[val_idx], masks[val_idx], num_support=num_support, target_size=target_size, seed=seed)
    test_dataset = InContextDataset(images[test_idx], masks[test_idx], num_support=num_support, target_size=target_size, seed=seed)

    split_sizes = {
        "total": total_slices,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx)
    }

    return train_dataset, val_dataset, test_dataset, split_sizes


def get_braintumour_splits(
    data_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_support: int = 2,
    target_size: tuple = (128, 128),
    channel_idx: int = 0
):
    """
    Loads Brain Tumour images (memory-mapped) and masks, splits into train/val/test sets,
    and creates InContextDataset instances.
    
    Parameters
    ----------
    channel_idx : int, optional
        If set (e.g. 0 for FLAIR), extracts a single channel from (N, H, W, 4) MRI volumes.
        If None, preserves all 4 MRI channels.
    """
    images_path = os.path.join(data_dir, "braintumour_images.npy")
    masks_path = os.path.join(data_dir, "braintumour_masks.npy")

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        raise FileNotFoundError(f"Processed braintumour data not found at {images_path} and {masks_path}")

    # Use memory mapping to avoid loading entire 29GB array into RAM
    images_mm = np.load(images_path, mmap_mode="r")  # (N, H, W, 4)
    masks_mm = np.load(masks_path, mmap_mode="r")    # (N, H, W)

    if channel_idx is not None:
        images = images_mm[:, :, :, channel_idx]     # (N, H, W)
    else:
        images = images_mm                           # (N, H, W, 4)
    masks = masks_mm

    total_slices = len(images)
    indices = np.arange(total_slices)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_end = int(total_slices * train_ratio)
    val_end = train_end + int(total_slices * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print("==================================================")
    print(" Data Split Statistics (MSD Task01 BrainTumour)")
    print("==================================================")
    print(f"  Total Slices: {total_slices}")
    print(f"  Train Set:    {len(train_idx)} slices ({len(train_idx)/total_slices*100:.1f}%)")
    print(f"  Val Set:      {len(val_idx)} slices ({len(val_idx)/total_slices*100:.1f}%)")
    print(f"  Test Set:     {len(test_idx)} slices ({len(test_idx)/total_slices*100:.1f}%)")
    print("--------------------------------------------------")

    train_dataset = InContextDataset(images[train_idx], masks[train_idx], num_support=num_support, target_size=target_size, seed=seed)
    val_dataset = InContextDataset(images[val_idx], masks[val_idx], num_support=num_support, target_size=target_size, seed=seed)
    test_dataset = InContextDataset(images[test_idx], masks[test_idx], num_support=num_support, target_size=target_size, seed=seed)

    split_sizes = {
        "total": total_slices,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx)
    }

    return train_dataset, val_dataset, test_dataset, split_sizes


if __name__ == "__main__":
    print("Testing Spleen Dataset Splits...")
    tr_s, val_s, te_s, _ = get_spleen_splits()
    sample = tr_s[0]
    print(f"  Query Img Shape:     {list(sample[0].shape)}")
    print(f"  Query Mask Shape:    {list(sample[1].shape)}")
    print(f"  Support Imgs Shape:  {list(sample[2].shape)}")
    print(f"  Support Masks Shape: {list(sample[3].shape)}")

    print("\nTesting Brain Tumour Dataset Splits (FLAIR single channel)...")
    tr_b, val_b, te_b, _ = get_braintumour_splits(channel_idx=0)
    sample_b = tr_b[0]
    print(f"  Query Img Shape:     {list(sample_b[0].shape)}")
    print(f"  Query Mask Shape:    {list(sample_b[1].shape)}")
    print(f"  Support Imgs Shape:  {list(sample_b[2].shape)}")
    print(f"  Support Masks Shape: {list(sample_b[3].shape)}")


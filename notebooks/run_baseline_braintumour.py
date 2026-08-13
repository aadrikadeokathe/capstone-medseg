import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add external/universeg to sys.path to import UniverSeg
sys.path.append(os.path.abspath("external/universeg"))
from universeg import universeg


def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Dice similarity coefficient between binary arrays pred and target.
    """
    intersection = np.sum(pred * target)
    total = np.sum(pred) + np.sum(target)
    return float((2.0 * intersection + eps) / (total + eps))


def run_baseline():
    images_path = "data/processed/braintumour_images.npy"
    masks_path = "data/processed/braintumour_masks.npy"

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        raise FileNotFoundError(f"Processed dataset not found at {images_path} and {masks_path}")

    # Load memory-mapped or numpy arrays
    print(f"Loading processed dataset from {images_path} and {masks_path}...")
    images_4d = np.load(images_path, mmap_mode="r")  # shape (N, H, W, 4)
    masks_3d = np.load(masks_path, mmap_mode="r")    # shape (N, H, W)

    dataset_shape = images_4d.shape
    slice_count = dataset_shape[0]
    print(f"Dataset Loaded Successfully:")
    print(f"  Images 4D shape: {dataset_shape}")
    print(f"  Masks 3D shape:  {masks_3d.shape}")
    print(f"  Total Slices:    {slice_count}")

    # LIMITATION NOTE:
    # UniverSeg expects single-channel (grayscale) 2D images, not 4-channel MRI tensors.
    # For this initial baseline test, we extract only the FLAIR modality (channel index 0),
    # which is commonly used alone for tumor visibility.
    # Note on limitation: full multimodal fusion (combining FLAIR, T1, T1gd, T2) is left as future work.
    # This baseline evaluates single-channel cross-modality generalization of UniverSeg.
    flair_images = images_4d[:, :, :, 0]  # shape (N, H, W)

    # Randomly select 2 support slices and 5 test slices
    rng = np.random.RandomState(42)
    selected_indices = rng.choice(slice_count, size=7, replace=False)
    support_indices = selected_indices[:2]
    test_indices = selected_indices[2:]

    print(f"Selected Support Set Slice Indices: {support_indices}")
    print(f"Selected Test Set Slice Indices:    {test_indices}")

    # Extract support and test arrays
    support_imgs = flair_images[support_indices]  # shape (2, H, W)
    support_lbls = masks_3d[support_indices]      # shape (2, H, W)

    test_imgs = flair_images[test_indices]        # shape (5, H, W)
    test_lbls = masks_3d[test_indices]            # shape (5, H, W)

    # Format tensors for UniverSeg inference:
    # target_image: (B, 1, H, W) where B=5
    # support_images: (B, S, 1, H, W) where B=5, S=2
    # support_labels: (B, S, 1, H, W) where B=5, S=2
    num_test = len(test_indices)
    num_support = len(support_indices)

    test_img_t = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1)  # (5, 1, H, W)
    
    support_img_t = torch.tensor(support_imgs, dtype=torch.float32).unsqueeze(1).unsqueeze(0).repeat(num_test, 1, 1, 1, 1)  # (5, 2, 1, H, W)
    support_lbl_t = torch.tensor(support_lbls, dtype=torch.float32).unsqueeze(1).unsqueeze(0).repeat(num_test, 1, 1, 1, 1)  # (5, 2, 1, H, W)

    # Load pretrained UniverSeg model
    print("Loading pretrained UniverSeg model...")
    model = universeg(pretrained=True)
    model.eval()

    print("Running UniverSeg inference on 5 test slices...")
    with torch.no_grad():
        logits = model(test_img_t, support_img_t, support_lbl_t)  # shape (5, 1, H, W)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()     # shape (5, H, W)
        preds = (probs > 0.5).astype(np.float32)

    # Compute Dice scores
    dice_scores = []
    for i in range(num_test):
        d_score = compute_dice(preds[i], test_lbls[i])
        dice_scores.append(d_score)
        print(f"  Test Slice {i+1} (Index {test_indices[i]}): Dice = {d_score:.4f}")

    avg_dice = float(np.mean(dice_scores))
    print(f"Average Dice Score across 5 test slices: {avg_dice:.4f}")

    # Ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    # Save scores log text file
    scores_log_path = "logs/braintumour_baseline_scores.txt"
    with open(scores_log_path, "w") as f:
        f.write("UniverSeg Brain Tumour Baseline Results\n")
        f.write("=======================================\n")
        f.write(f"Dataset Shape (4D Images): {dataset_shape}\n")
        f.write(f"Dataset Shape (3D Masks):  {masks_3d.shape}\n")
        f.write(f"Support Slice Indices: {list(support_indices)}\n")
        f.write(f"Test Slice Indices:    {list(test_indices)}\n\n")
        for i, d in enumerate(dice_scores):
            f.write(f"Test Slice {i+1} (Index {test_indices[i]}): Dice = {d:.4f}\n")
        f.write(f"\nAverage Dice Score: {avg_dice:.4f}\n")

    print(f"Saved scores log to {scores_log_path}")

    # Plot visual comparison (display all 5 test slices side-by-side)
    fig, axes = plt.subplots(num_test, 3, figsize=(12, 4 * num_test))
    fig.suptitle("UniverSeg Brain Tumour Baseline Results (FLAIR Channel)", fontsize=16, y=0.99)

    for i in range(num_test):
        # FLAIR slice
        axes[i, 0].imshow(test_imgs[i], cmap="gray")
        axes[i, 0].set_title(f"Test Slice {i+1} - FLAIR Input")
        axes[i, 0].axis("off")

        # Ground Truth Mask
        axes[i, 1].imshow(test_lbls[i], cmap="bone")
        axes[i, 1].set_title(f"Ground Truth Mask (Slice {test_indices[i]})")
        axes[i, 1].axis("off")

        # Predicted Mask
        axes[i, 2].imshow(preds[i], cmap="bone")
        axes[i, 2].set_title(f"UniverSeg Prediction (Dice: {dice_scores[i]:.4f})")
        axes[i, 2].axis("off")

    plt.tight_layout()
    results_plot_path = "logs/braintumour_baseline_results.png"
    plt.savefig(results_plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved visual results plot to {results_plot_path}")

    return dataset_shape, dice_scores, avg_dice, results_plot_path, scores_log_path


if __name__ == "__main__":
    run_baseline()

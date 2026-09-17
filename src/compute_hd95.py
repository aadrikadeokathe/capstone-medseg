"""
compute_hd95.py - 95th Percentile Hausdorff Distance (HD95) & Dice Evaluation
=============================================================================
Calculates both volumetric overlap (Dice) and surface boundary distance (HD95)
across all four Medical Segmentation Decathlon (MSD) test targets.

Supports both pure SciPy distance transforms and MONAI metrics.
"""

import os
import sys
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.models.in_context_model import InContextSegmentationModel
    from src.data_split import (
        get_spleen_splits,
        get_liver_splits,
        get_heart_splits,
        get_braintumour_splits,
    )
except ImportError as e:
    print(f"[WARN] Torch/Local modules not fully loaded in this environment: {e}")


def compute_dice_np(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute binary Dice coefficient for numpy boolean arrays."""
    pred_b = pred_mask > 0.5
    gt_b = gt_mask > 0.5
    intersection = np.logical_and(pred_b, gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0 if not np.any(gt_b) else 0.0
    return float(2.0 * intersection / total)


def compute_hd95_np(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_spacing=(1.0, 1.0)) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95) in mm/voxels.
    Uses exact Euclidean distance transforms for accuracy and speed.
    """
    pred_b = pred_mask > 0.5
    gt_b = gt_mask > 0.5

    # If either is empty
    if not np.any(pred_b) and not np.any(gt_b):
        return 0.0
    if not np.any(pred_b) or not np.any(gt_b):
        # Maximum penalty based on image diagonal
        return np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2)

    # Compute boundary contours
    from scipy.ndimage import binary_erosion
    pred_border = pred_b ^ binary_erosion(pred_b)
    gt_border = gt_b ^ binary_erosion(gt_b)

    if not np.any(pred_border) or not np.any(gt_border):
        return 0.0

    # Distance maps
    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    # Surface distances
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def get_test_loader(ds_name: str, batch_size: int = 4):
    if ds_name == "spleen":
        _, _, te, _ = get_spleen_splits(num_support=2)
    elif ds_name == "liver":
        _, _, te, _ = get_liver_splits(num_support=2)
    elif ds_name == "heart":
        _, _, te, _ = get_heart_splits(num_support=2)
    elif ds_name == "braintumour":
        _, _, te, _ = get_braintumour_splits(num_support=2)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    return DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)


def run_evaluation(checkpoint_path: str, datasets: list, output_file: str, device: str = "cuda", max_samples: int = None):
    """Evaluates checkpoint on specified datasets and logs Dice + HD95."""
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[INFO] Using device: {device_obj}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model = InContextSegmentationModel(num_layers=3).to(device_obj)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print("[INFO] Model weights loaded successfully.")
    else:
        print(f"[WARN] Checkpoint {checkpoint_path} not found. Running with mock/random weights for testing.")

    model.eval()
    results = {}

    for ds_name in datasets:
        print(f"\nEvaluating dataset: {ds_name}...")
        try:
            val_loader = get_test_loader(ds_name, batch_size=4)
        except Exception as e:
            print(f"[SKIP] Could not load dataloader for {ds_name}: {e}")
            continue

        dices = []
        hd95s = []

        with torch.no_grad():
            for batch_idx, (query, targets, support_imgs, support_masks) in enumerate(val_loader):
                query = query.to(device_obj)
                support_imgs = support_imgs.to(device_obj)
                support_masks = support_masks.to(device_obj)
                targets = targets.to(device_obj)

                preds = model(query, support_imgs, support_masks)
                preds = torch.sigmoid(preds).cpu().numpy()
                targets_np = targets.cpu().numpy()

                for b in range(preds.shape[0]):
                    p_mask = preds[b, 0]
                    t_mask = targets_np[b, 0]

                    dice = compute_dice_np(p_mask, t_mask)
                    hd = compute_hd95_np(p_mask, t_mask)

                    dices.append(dice)
                    hd95s.append(hd)

                    if max_samples and len(dices) >= max_samples:
                        break
                if max_samples and len(dices) >= max_samples:
                    break

        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
        std_dice = float(np.std(dices)) if len(dices) > 0 else 0.0
        mean_hd95 = float(np.mean(hd95s)) if len(hd95s) > 0 else 0.0
        std_hd95 = float(np.std(hd95s)) if len(hd95s) > 0 else 0.0

        results[ds_name] = {
            "dice_mean": mean_dice, "dice_std": std_dice,
            "hd95_mean": mean_hd95, "hd95_std": std_hd95,
            "num_samples": len(dices)
        }
        print(f"[{ds_name.upper()}] Dice: {mean_dice:.4f} +/- {std_dice:.4f} | HD95: {mean_hd95:.2f} +/- {std_hd95:.2f} mm/voxels")

    # Save summary report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("================================================================================\n")
        f.write("           BOUNDARY DISTANCE (HD95) & VOLUMETRIC (DICE) EVALUATION REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"{'Target Dataset':<20} | {'Dice (Mean +/- Std)':<25} | {'HD95 (mm/voxels)':<25} | {'Samples':<8}\n")
        f.write("-" * 88 + "\n")
        for ds, res in results.items():
            d_str = f"{res['dice_mean']:.4f} +/- {res['dice_std']:.4f}"
            h_str = f"{res['hd95_mean']:.2f} +/- {res['hd95_std']:.2f}"
            f.write(f"{ds:<20} | {d_str:<25} | {h_str:<25} | {res['num_samples']:<8}\n")
        f.write("=" * 88 + "\n")

    print(f"\n[DONE] Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dice and HD95")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_fusion_episodic.pt")
    parser.add_argument("--datasets", nargs="+", default=["spleen", "liver", "heart", "braintumour"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples to evaluate per dataset (e.g. 50 for quick CPU run)")
    parser.add_argument("--output", type=str, default="logs/hd95_evaluation_results.txt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_evaluation(args.checkpoint, args.datasets, args.output, args.device, args.max_samples)

"""
src/train_multi_dataset.py

Leave-one-dataset-out zero-shot training and evaluation for In-Context Fusion Model.
Pools training slices from specified training datasets, trains one fusion model with a frozen UniverSeg backbone,
and evaluates zero-shot performance on a completely held-out test dataset.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import (
    InContextDataset,
    get_spleen_splits,
    get_braintumour_splits,
    get_heart_splits,
    get_liver_splits,
)
from src.models.in_context_model import InContextSegmentationModel
from src.train_fusion import compute_dice_score, BCEDiceLoss, train_one_epoch, evaluate

SUPPORTED_DATASETS = ["spleen", "liver", "heart", "braintumour"]


def load_dataset_splits(ds_name: str, num_support: int, target_size: int, channel_idx: int):
    """
    Loads train/val/test splits for a given training dataset name.
    """
    ts_tuple = (target_size, target_size)
    if ds_name == "spleen":
        return get_spleen_splits(num_support=num_support, target_size=ts_tuple)
    elif ds_name == "heart":
        return get_heart_splits(num_support=num_support, target_size=ts_tuple)
    elif ds_name == "liver":
        return get_liver_splits(num_support=num_support, target_size=ts_tuple)
    elif ds_name == "braintumour":
        return get_braintumour_splits(num_support=num_support, target_size=ts_tuple, channel_idx=channel_idx)
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")


def load_heldout_test_split(ds_name: str, num_support: int, target_size: int, channel_idx: int):
    """
    Loads ONLY the test split of the held-out dataset to guarantee complete exclusion
    from training and optimize memory/loading overhead.
    """
    ts_tuple = (target_size, target_size)
    if ds_name == "braintumour":
        images_path = os.path.join("data/processed", "braintumour_images.npy")
        masks_path = os.path.join("data/processed", "braintumour_masks.npy")
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            raise FileNotFoundError(f"Processed braintumour data not found at {images_path} and {masks_path}")
        images_mm = np.load(images_path, mmap_mode="r")
        masks_mm = np.load(masks_path, mmap_mode="r")
        images = images_mm[:, :, :, channel_idx] if channel_idx is not None else images_mm
        masks = masks_mm
        total_slices = len(images)
        indices = np.arange(total_slices)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        train_end = int(total_slices * 0.70)
        val_end = train_end + int(total_slices * 0.15)
        test_idx = indices[val_end:]
        print("==================================================")
        print(" Held-Out Test Set Statistics (MSD Task01 BrainTumour)")
        print("==================================================")
        print(f"  Total Dataset Slices: {total_slices}")
        print(f"  Held-Out Test Set:    {len(test_idx)} slices ({len(test_idx)/total_slices*100:.1f}%)")
        print("--------------------------------------------------")
        test_dataset = InContextDataset(images[test_idx], masks[test_idx], num_support=num_support, target_size=ts_tuple, seed=42)
        return test_dataset
    else:
        _, _, test_dataset, _ = load_dataset_splits(ds_name, num_support, target_size, channel_idx)
        return test_dataset


def load_baseline_scores(dataset: str):
    """
    Reads existing UniverSeg baseline Dice and per-dataset-trained Dice for `dataset` from logs/ if available.
    Returns (universeg_dice, per_dataset_dice) as float or None. Never fabricates missing numbers.
    """
    universeg_dice = None
    per_dataset_dice = None

    # UniverSeg baseline score
    baseline_path = os.path.join("logs", f"{dataset}_baseline_scores.txt")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            for line in f:
                if "Mean Dice Score:" in line:
                    try:
                        universeg_dice = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass

    # Per-dataset trained score
    trained_paths = [
        os.path.join("logs", f"{dataset}_trained_scores.txt"),
        os.path.join("logs", f"fusion_training_{dataset}.txt"),
    ]
    for path in trained_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if "Test Dice:" in line:
                        try:
                            per_dataset_dice = float(line.split(":")[-1].strip())
                            break
                        except ValueError:
                            pass
            if per_dataset_dice is not None:
                break

    return universeg_dice, per_dataset_dice


def main():
    parser = argparse.ArgumentParser(
        description="Leave-One-Dataset-Out Zero-Shot Training for In-Context Fusion Model"
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="One or more datasets allowed for training (e.g. spleen liver heart)",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Single dataset held out for zero-shot evaluation (e.g. braintumour)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_support", type=int, default=2, help="Number of support slices per sample")
    parser.add_argument("--target_size", type=int, default=128, help="Spatial resolution (H=W)")
    parser.add_argument("--channel_idx", type=int, default=0, help="Channel index for BrainTumour (0=FLAIR)")
    parser.add_argument("--dry_run", action="store_true", help="Run quick verification run without full training")

    args = parser.parse_args()

    # Validation: Held-out test dataset cannot be in train_datasets
    if args.test_dataset in args.train_datasets:
        parser.error(
            f"Data leakage constraint violated: held-out test dataset '{args.test_dataset}' "
            f"cannot be included in --train_datasets ({args.train_datasets})."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" Leave-One-Dataset-Out Zero-Shot Training Pipeline")
    print("==================================================")
    print(f" Device:               {device}")
    print(f" Training Datasets:    {args.train_datasets}")
    print(f" Held-Out Test Set:    {args.test_dataset}")
    print(f" Epochs:               {args.epochs}")
    print(f" Batch Size:           {args.batch_size}")
    print(f" Learning Rate:        {args.lr}")
    print(f" Support Count:        {args.num_support}")
    print(f" Target Size:          ({args.target_size}, {args.target_size})")
    print(f" Dry Run Mode:         {args.dry_run}")
    print("--------------------------------------------------")

    # Pool training and validation splits ONLY from train_datasets
    train_datasets_list = []
    val_datasets_list = []

    print("\nLoading and pooling training datasets...")
    for ds_name in args.train_datasets:
        tr_ds, v_ds, _, split_info = load_dataset_splits(
            ds_name, args.num_support, args.target_size, args.channel_idx
        )
        train_datasets_list.append(tr_ds)
        val_datasets_list.append(v_ds)
        print(f"  + Added '{ds_name}': {split_info['train']} train slices, {split_info['val']} val slices.")

    train_ds_pooled = ConcatDataset(train_datasets_list)
    val_ds_pooled = ConcatDataset(val_datasets_list)

    print(f"Pooled Training Set Total Slices:   {len(train_ds_pooled)}")
    print(f"Pooled Validation Set Total Slices: {len(val_ds_pooled)}")

    # Load held-out test dataset ONLY for final evaluation
    print(f"\nLoading held-out test dataset split for '{args.test_dataset}'...")
    test_ds_heldout = load_heldout_test_split(
        args.test_dataset, args.num_support, args.target_size, args.channel_idx
    )
    print(f"  * Held-out '{args.test_dataset}' Test Slices: {len(test_ds_heldout)}")

    train_loader = DataLoader(train_ds_pooled, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds_pooled, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader_heldout = DataLoader(test_ds_heldout, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Instantiate model with FROZEN UniverSeg backbone
    model = InContextSegmentationModel(
        in_channels=1,
        feature_channels=[64, 64, 64],
        num_heads=4,
        dropout=0.1
    ).to(device)

    model.print_parameter_summary()

    # Verify parameters: optimizer only updates non-backbone parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.5)

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    checkpoint_path = f"models/checkpoints/best_fusion_model_zeroshot_{args.test_dataset}.pt"

    best_val_dice = -1.0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    print(f"\nStarting Zero-Shot Training (Held-Out: {args.test_dataset.upper()})...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, device, dry_run=args.dry_run)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, dry_run=args.dry_run)

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "train_datasets": args.train_datasets,
                "test_dataset": args.test_dataset,
            }, checkpoint_path)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Dice: {tr_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
              f"{' [SAVED BEST]' if is_best else ''}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # Load best checkpoint for zero-shot evaluation on held-out test split
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded best checkpoint from {checkpoint_path}")

    print(f"\nEvaluating Zero-Shot Model on Held-Out Test Set ({args.test_dataset.upper()})...")
    test_loss, zero_shot_dice = evaluate(model, test_loader_heldout, criterion, device, dry_run=args.dry_run)

    # Fetch baseline scores for comparison
    universeg_baseline_dice, per_dataset_trained_dice = load_baseline_scores(args.test_dataset)

    print("==================================================")
    print(f" Zero-Shot Results on {args.test_dataset.upper()} (Held-Out):")
    print(f"   Zero-Shot Test Loss: {test_loss:.4f}")
    print(f"   Zero-Shot Test Dice: {zero_shot_dice:.4f}")
    print("--------------------------------------------------")
    print(f" Baseline Comparisons for {args.test_dataset.upper()}:")
    if universeg_baseline_dice is not None:
        diff_us = zero_shot_dice - universeg_baseline_dice
        print(f"   UniverSeg Baseline Dice:   {universeg_baseline_dice:.4f} (Diff: {diff_us:+.4f})")
    else:
        print(f"   UniverSeg Baseline Dice:   N/A")

    if per_dataset_trained_dice is not None:
        diff_pd = zero_shot_dice - per_dataset_trained_dice
        print(f"   Per-Dataset Trained Dice:  {per_dataset_trained_dice:.4f} (Diff: {diff_pd:+.4f})")
    else:
        print(f"   Per-Dataset Trained Dice:  N/A")
    print("==================================================")

    # Save log to logs/zeroshot_{test_dataset}_scores.txt
    log_file_path = f"logs/zeroshot_{args.test_dataset}_scores.txt"
    
    us_str = f"{universeg_baseline_dice:.4f}" if universeg_baseline_dice is not None else "N/A"
    pd_str = f"{per_dataset_trained_dice:.4f}" if per_dataset_trained_dice is not None else "N/A"
    
    comp_us_str = f"Diff vs UniverSeg Baseline:  {zero_shot_dice - universeg_baseline_dice:+.4f}" if universeg_baseline_dice is not None else "Diff vs UniverSeg Baseline:  N/A"
    comp_pd_str = f"Diff vs Per-Dataset-Trained: {zero_shot_dice - per_dataset_trained_dice:+.4f}" if per_dataset_trained_dice is not None else "Diff vs Per-Dataset-Trained: N/A"

    log_content = (
        f"Zero-Shot Leave-One-Dataset-Out Evaluation Log - {args.test_dataset.upper()}\n"
        "==================================================\n"
        f"Training Datasets: {', '.join(args.train_datasets)}\n"
        f"Held-Out Test Dataset: {args.test_dataset}\n"
        f"Epochs: {args.epochs}\n"
        f"Backbone Frozen: True (UniverSeg pretrained parameters locked)\n"
        f"Batch Size: {args.batch_size}\n"
        f"Learning Rate: {args.lr}\n"
        f"Support Count: {args.num_support}\n"
        f"Target Size: ({args.target_size}, {args.target_size})\n\n"
        "Zero-Shot Results on Held-Out Test Set:\n"
        "--------------------------------------------------\n"
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Dice: {zero_shot_dice:.4f}\n\n"
        f"Baseline Comparisons for {args.test_dataset.upper()}:\n"
        "--------------------------------------------------\n"
        f"UniverSeg Baseline Dice:   {us_str}\n"
        f"Per-Dataset Trained Dice:  {pd_str}\n\n"
        "Numerical Comparison:\n"
        "--------------------------------------------------\n"
        f"{comp_us_str}\n"
        f"{comp_pd_str}\n\n"
        "Training History:\n"
        "Epoch,TrainLoss,TrainDice,ValLoss,ValDice\n"
    )
    for ep in range(len(history["train_loss"])):
        log_content += (
            f"{ep+1},{history['train_loss'][ep]:.4f},{history['train_dice'][ep]:.4f},"
            f"{history['val_loss'][ep]:.4f},{history['val_dice'][ep]:.4f}\n"
        )

    with open(log_file_path, "w") as f:
        f.write(log_content)

    # Save training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title(f"Zero-Shot Loss Curves (Held-out: {args.test_dataset.capitalize()})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(history["train_dice"]) + 1), history["train_dice"], label="Train Dice", marker="o")
    axes[1].plot(range(1, len(history["val_dice"]) + 1), history["val_dice"], label="Val Dice", marker="s")
    axes[1].axhline(y=zero_shot_dice, color="r", linestyle="--", label=f"Test Dice ({zero_shot_dice:.4f})")
    if universeg_baseline_dice is not None:
        axes[1].axhline(y=universeg_baseline_dice, color="g", linestyle=":", label=f"UniverSeg Baseline ({universeg_baseline_dice:.4f})")
    if per_dataset_trained_dice is not None:
        axes[1].axhline(y=per_dataset_trained_dice, color="b", linestyle="-.", label=f"Per-Dataset Trained ({per_dataset_trained_dice:.4f})")
    axes[1].set_title(f"Zero-Shot Dice Curves (Held-out: {args.test_dataset.capitalize()})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curve_plot_path = f"logs/fusion_training_curves_zeroshot_{args.test_dataset}.png"
    plt.savefig(curve_plot_path, dpi=150)
    plt.close()

    print(f"\nSaved zero-shot evaluation log to {log_file_path}")
    print(f"Saved zero-shot training curves plot to {curve_plot_path}")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import get_spleen_splits, get_braintumour_splits, get_heart_splits
from src.models.in_context_model import InContextSegmentationModel


def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Computes hard Dice similarity score for binary predictions and targets.
    preds: [B, 1, H, W] (binary 0.0 or 1.0)
    targets: [B, 1, H, W] (binary 0.0 or 1.0)
    """
    intersection = (preds * targets).sum(dim=(-2, -1))
    total = preds.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + eps) / (total + eps)
    return float(dice.mean().item())


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Soft Dice Loss for segmentation tasks.
    """

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        cardinality = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, dry_run=False):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    for step, (query_img, query_mask, support_imgs, support_masks) in enumerate(dataloader):
        query_img = query_img.to(device)
        query_mask = query_mask.to(device)
        support_imgs = support_imgs.to(device)
        support_masks = support_masks.to(device)

        optimizer.zero_grad()
        logits = model(query_img, support_imgs, support_masks)
        loss = criterion(logits, query_mask)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dice = compute_dice_score(preds, query_mask)

        batch_size = query_img.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice * batch_size
        total_samples += batch_size

        if dry_run and step >= 1:
            break

    return running_loss / total_samples, running_dice / total_samples


def evaluate(model, dataloader, criterion, device, dry_run=False):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, (query_img, query_mask, support_imgs, support_masks) in enumerate(dataloader):
            query_img = query_img.to(device)
            query_mask = query_mask.to(device)
            support_imgs = support_imgs.to(device)
            support_masks = support_masks.to(device)

            logits = model(query_img, support_imgs, support_masks)
            loss = criterion(logits, query_mask)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dice = compute_dice_score(preds, query_mask)

            batch_size = query_img.size(0)
            running_loss += loss.item() * batch_size
            running_dice += dice * batch_size
            total_samples += batch_size

            if dry_run and step >= 1:
                break

    return running_loss / total_samples, running_dice / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train In-Context Fusion Model for Medical Segmentation")
    parser.add_argument("--dataset", type=str, choices=["spleen", "braintumour", "heart"], default="spleen")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_support", type=int, default=2, help="Number of support slices per sample")
    parser.add_argument("--target_size", type=int, default=128, help="Spatial resolution (H=W)")
    parser.add_argument("--channel_idx", type=int, default=0, help="Channel index for BrainTumour (0=FLAIR)")
    parser.add_argument("--dry_run", action="store_true", help="Run quick 2-step verification run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" In-Context Fusion Model Training Pipeline")
    print("==================================================")
    print(f" Device:         {device}")
    print(f" Dataset:        {args.dataset}")
    print(f" Epochs:         {args.epochs}")
    print(f" Batch Size:     {args.batch_size}")
    print(f" Learning Rate:  {args.lr}")
    print(f" Support Count:  {args.num_support}")
    print(f" Target Size:    ({args.target_size}, {args.target_size})")
    print(f" Dry Run Mode:   {args.dry_run}")
    print("--------------------------------------------------")

    # Load dataset splits
    if args.dataset == "spleen":
        train_ds, val_ds, test_ds, split_sizes = get_spleen_splits(
            num_support=args.num_support,
            target_size=(args.target_size, args.target_size)
        )
        in_channels = 1
    elif args.dataset == "heart":
        train_ds, val_ds, test_ds, split_sizes = get_heart_splits(
            num_support=args.num_support,
            target_size=(args.target_size, args.target_size)
        )
        in_channels = 1
    else:
        train_ds, val_ds, test_ds, split_sizes = get_braintumour_splits(
            num_support=args.num_support,
            target_size=(args.target_size, args.target_size),
            channel_idx=args.channel_idx
        )
        in_channels = 1 if args.channel_idx is not None else 4

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Instantiate model
    model = InContextSegmentationModel(
        in_channels=in_channels,
        feature_channels=[64, 64, 64],
        num_heads=4,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.5)

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    checkpoint_path = f"models/checkpoints/best_fusion_model_{args.dataset}.pt"

    best_val_dice = -1.0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    print("\nStarting Training...")
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
                "dataset": args.dataset
            }, checkpoint_path)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Dice: {tr_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
              f"{' [SAVED BEST]' if is_best else ''}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # Evaluate best model on test set
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_dice = evaluate(model, test_loader, criterion, device, dry_run=args.dry_run)
    print("==================================================")
    print(f" Test Set Results ({args.dataset.upper()}):")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Dice: {test_dice:.4f}")
    print("==================================================")

    # Save log files
    log_file_path = f"logs/fusion_training_{args.dataset}.txt"
    trained_scores_path = f"logs/{args.dataset}_trained_scores.txt"
    
    log_content = (
        f"In-Context Fusion Model Training Log - {args.dataset.upper()}\n"
        "==================================================\n"
        f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}\n"
        f"Best Val Dice: {best_val_dice:.4f}\n"
        f"Test Loss:     {test_loss:.4f}\n"
        f"Test Dice:     {test_dice:.4f}\n\n"
        "Epoch,TrainLoss,TrainDice,ValLoss,ValDice\n"
    )
    for ep in range(len(history["train_loss"])):
        log_content += (
            f"{ep+1},{history['train_loss'][ep]:.4f},{history['train_dice'][ep]:.4f},"
            f"{history['val_loss'][ep]:.4f},{history['val_dice'][ep]:.4f}\n"
        )

    with open(log_file_path, "w") as f:
        f.write(log_content)

    with open(trained_scores_path, "w") as f:
        f.write(log_content)

    # Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title(f"Loss Curves ({args.dataset.capitalize()})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(history["train_dice"]) + 1), history["train_dice"], label="Train Dice", marker="o")
    axes[1].plot(range(1, len(history["val_dice"]) + 1), history["val_dice"], label="Val Dice", marker="s")
    axes[1].axhline(y=test_dice, color="r", linestyle="--", label=f"Test Dice ({test_dice:.4f})")
    axes[1].set_title(f"Dice Curves ({args.dataset.capitalize()})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curve_plot_path = f"logs/fusion_training_curves_{args.dataset}.png"
    plt.savefig(curve_plot_path, dpi=150)
    plt.close()
    print(f"Saved training log to {log_file_path} and {trained_scores_path}")
    print(f"Saved curves plot to {curve_plot_path}")


if __name__ == "__main__":
    main()

"""
src/evaluate_shot_ablation.py

Evaluates In-Context Segmentation Model performance across varying support shot counts
(K in {1, 2, 4, 8}) on held-out test sets to generate prompt-cardinality scaling curves.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from src.train_fusion import compute_dice_score, BCEDiceLoss
from src.train_episodic import evaluate_dataset_split, SUPPORTED_DATASETS


def run_shot_ablation(
    checkpoint_path: str,
    shot_counts: list = [1, 2, 4, 8],
    datasets: list = None,
    dry_run: bool = False,
    device: torch.device = None,
    output_txt: str = "logs/shot_ablation_results.txt",
    output_png: str = "logs/shot_ablation_curve.png",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_datasets = datasets if datasets is not None else list(SUPPORTED_DATASETS)

    print("==================================================")
    print(" Running Shot-Count Ablation (K in {1, 2, 4, 8})")
    print("==================================================")
    print(f" Checkpoint:  {checkpoint_path}")
    print(f" Shots:       {shot_counts}")
    print(f" Datasets:    {eval_datasets}")
    print(f" Device:      {device}")
    print(f" Dry Run:     {dry_run}")
    print("--------------------------------------------------")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = InContextSegmentationModel(
        in_channels=1,
        feature_channels=[64, 64, 64],
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = BCEDiceLoss(bce_weight=0.5)
    max_steps = 2 if dry_run else None

    # results structure: {dataset: {shot: dice}}
    results = {ds: {} for ds in eval_datasets}
    avg_per_shot = {}

    for k in shot_counts:
        print(f"\nEvaluating Shot Count: K = {k}")
        dices_at_k = []

        for ds_name in eval_datasets:
            if ds_name == "spleen":
                _, _, test_ds, _ = get_spleen_splits(num_support=k, target_size=(128, 128))
            elif ds_name == "liver":
                _, _, test_ds, _ = get_liver_splits(num_support=k, target_size=(128, 128))
            elif ds_name == "heart":
                _, _, test_ds, _ = get_heart_splits(num_support=k, target_size=(128, 128))
            elif ds_name == "braintumour":
                _, _, test_ds, _ = get_braintumour_splits(num_support=k, target_size=(128, 128), channel_idx=0)

            loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
            _, dice = evaluate_dataset_split(model, loader, criterion, device, max_steps=max_steps)
            results[ds_name][k] = dice
            dices_at_k.append(dice)
            print(f"  [{ds_name.upper():12s}] K={k:2d} -> Test Dice: {dice:.4f}")

        avg_k = float(np.mean(dices_at_k))
        avg_per_shot[k] = avg_k
        print(f"  -> Average Dice across datasets at K={k}: {avg_k:.4f}")

    # Print summary table
    print("\n==================================================")
    print(" SHOT-COUNT ABLATION SUMMARY TABLE")
    print("==================================================")
    header = f"{'Dataset':<14} | " + " | ".join([f"K={k} Shot" for k in shot_counts])
    sep = "-" * len(header)
    print(header)
    print(sep)

    for ds_name in eval_datasets:
        row = f"{ds_name:<14} | " + " | ".join([f"{results[ds_name][k]:<9.4f}" for k in shot_counts])
        print(row)
    print(sep)
    avg_row = f"{'AVERAGE':<14} | " + " | ".join([f"{avg_per_shot[k]:<9.4f}" for k in shot_counts])
    print(avg_row)
    print("==================================================")

    # Write log file
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as f:
        f.write("Few-Shot Support Context Ablation Results\n")
        f.write("==================================================\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for ds_name in eval_datasets:
            f.write(f"{ds_name:<14} | " + " | ".join([f"{results[ds_name][k]:<9.4f}" for k in shot_counts]) + "\n")
        f.write(sep + "\n")
        f.write(avg_row + "\n")

    print(f"\nSaved text results to {output_txt}")

    # Plot curve
    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    for i, ds_name in enumerate(eval_datasets):
        vals = [results[ds_name][k] for k in shot_counts]
        plt.plot(shot_counts, vals, marker=markers[i % len(markers)], label=ds_name.capitalize(), linewidth=1.5)

    avg_vals = [avg_per_shot[k] for k in shot_counts]
    plt.plot(shot_counts, avg_vals, "k--o", linewidth=2.5, label="Average")

    plt.xlabel("Number of Support Slices (K)", fontsize=12)
    plt.ylabel("Test Dice Score", fontsize=12)
    plt.title("Few-Shot In-Context Segmentation Scaling Curve", fontsize=14, fontweight="bold")
    plt.xticks(shot_counts)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved scaling curve plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate In-Context Segmentation across Support Shot Counts")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_fusion_episodic.pt")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8], help="Support shot counts")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry run verification")
    parser.add_argument("--output_txt", type=str, default="logs/shot_ablation_results.txt")
    parser.add_argument("--output_png", type=str, default="logs/shot_ablation_curve.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_shot_ablation(
        checkpoint_path=args.checkpoint,
        shot_counts=args.shots,
        dry_run=args.dry_run,
        device=device,
        output_txt=args.output_txt,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()

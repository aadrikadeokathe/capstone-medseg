"""
src/train_episodic.py

Episodic Meta-Learning Training & Zero-Shot Evaluation Pipeline for Medical Image Segmentation.

========================================================================================
THEORETICAL FOUNDATION & METHODS DESCRIPTION (FOR PAPER / REPORT)
========================================================================================

1. Why Single-Organ & Naive Multi-Task Training Cause Zero-Shot Generalization Failure:
   - Organ Memorization: Standard training on a single organ (e.g., Spleen) causes the
     Cross-Attention FusionModule and Decoder to learn invariant anatomical spatial priors
     (e.g., "spleen is located in the left upper abdomen with smooth, crescent boundaries").
     As a result, the cross-attention mechanism collapses: it no longer dynamically aligns
     support examples to query features, but instead functions as a fixed feature detector.
   - Domain & Modality Imbalance: In a naive pooled multi-dataset setup, MSD BrainTumour
     contains 31,527 slices while MSD Liver contains only 57 slices. BrainTumour would
     dominate >95% of mini-batches, biasing the model entirely toward multi-modal brain MRI
     while ignoring CT abdominal organs.

2. The Episodic Meta-Learning Solution:
   - Task Formulation: We cast few-shot medical image segmentation as episodic meta-learning.
     At each training step (episode), a task dataset D_k is sampled with UNIFORM probability:
         P(D_k) = 1 / |Datasets| = 0.25 (for Spleen, Liver, Heart, BrainTumour)
     This decouples task representation from slice counts, ensuring Liver (57 slices) receives
     equal optimization frequency to BrainTumour (31,527 slices).
   - Episodic In-Context Conditioning: Within each episode, a support set S = {(x_s, y_s)}_{s=1}^K
     and an independent query slice (x_q, y_q) are sampled from D_k. The network is tasked with:
         "Given support examples (x_s, y_s) showing the target structure, segment the
          corresponding structure in query x_q."
   - Forcing Meta-In-Context Alignment: Because the organ identity, contrast, and imaging
     modality (CT vs. MRI) alternate randomly from step to step, the model cannot rely on
     memorizing organ-specific shapes or positions. It is forced to learn generic
     meta-in-context alignment: cross-attending to shared semantic representations between
     query tokens and mask-conditioned support tokens.
   - Zero-Shot Generalization: When evaluated on an unseen held-out organ, the model applies
     its acquired in-context matching capability to segment the novel structure purely from
     the provided support slices without weight updates.
========================================================================================
"""

import os
import sys
import time

print("[1/5] Starting train_episodic.py...", flush=True)
print("[2/5] Loading PyTorch & CUDA libraries (first load on Windows can take 15-30s)...", flush=True)

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

print(f"      -> PyTorch v{torch.__version__} loaded successfully.", flush=True)
print(f"      -> CUDA Available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"      -> GPU Device: {torch.cuda.get_device_name(0)}", flush=True)


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
from src.models.backbone import load_universeg_backbone, get_frozen_features
from src.train_fusion import compute_dice_score, BCEDiceLoss

SUPPORTED_DATASETS = ["spleen", "liver", "heart", "braintumour"]


class EpisodicMedSegDataset(Dataset):
    """
    Episodic Multi-Dataset Loader for Meta-Learning Medical Image Segmentation.

    Loads all MSD datasets (Spleen, Liver, Heart, BrainTumour) and provides
    uniform dataset sampling across episodes to prevent dominant datasets
    (e.g., BrainTumour: 31,527 slices) from drowning out small datasets
    (e.g., Liver: 57 slices).
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        num_support: int = 2,
        target_size: tuple = (128, 128),
        channel_idx: int = 0,
        seed: int = 42,
        datasets: list = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_support = num_support
        self.target_size = target_size
        self.channel_idx = channel_idx
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.dataset_names = datasets if datasets is not None else list(SUPPORTED_DATASETS)
        self.splits = {}

        print("==================================================")
        print(" Initializing EpisodicMedSegDataset")
        print("==================================================")

        for ds_name in self.dataset_names:
            if ds_name == "spleen":
                tr, val, te, sizes = get_spleen_splits(
                    data_dir=data_dir, num_support=num_support, target_size=target_size, seed=seed
                )
            elif ds_name == "liver":
                tr, val, te, sizes = get_liver_splits(
                    data_dir=data_dir, num_support=num_support, target_size=target_size, seed=seed
                )
            elif ds_name == "heart":
                tr, val, te, sizes = get_heart_splits(
                    data_dir=data_dir, num_support=num_support, target_size=target_size, seed=seed
                )
            elif ds_name == "braintumour":
                tr, val, te, sizes = get_braintumour_splits(
                    data_dir=data_dir,
                    num_support=num_support,
                    target_size=target_size,
                    channel_idx=channel_idx,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown dataset: {ds_name}")

            self.splits[ds_name] = {
                "train": tr,
                "val": val,
                "test": te,
                "sizes": sizes,
            }
            print(f"  [{ds_name.upper():12s}] Train: {sizes['train']:5d} | Val: {sizes['val']:5d} | Test: {sizes['test']:5d}")

        self.total_train_slices = sum(self.splits[d]["sizes"]["train"] for d in self.dataset_names)
        print("--------------------------------------------------")
        print(f" Total Train Slices Across Datasets: {self.total_train_slices:,}")
        print(f" Sampling Policy: UNIFORM DATASET PROBABILITY (1/{len(self.dataset_names)} per episode)")
        print("==================================================")

    def __len__(self) -> int:
        return self.total_train_slices

    def sample_episode(self, split: str = "train"):
        """
        Samples a single episodic meta-learning task:
          1. Uniformly samples one dataset (organ).
          2. From that dataset's split, randomly samples 1 query slice.
          3. Randomly samples num_support distinct slices as support context.
          4. Returns (query_img, query_mask, support_imgs, support_masks, dataset_name).
        """
        ds_name = self.rng.choice(self.dataset_names)
        split_ds: InContextDataset = self.splits[ds_name][split]
        total_slices = len(split_ds)

        query_idx = int(self.rng.choice(total_slices))
        available_support = [i for i in range(total_slices) if i != query_idx]

        if len(available_support) < self.num_support:
            support_indices = available_support
        else:
            support_indices = list(
                self.rng.choice(available_support, size=self.num_support, replace=False)
            )

        q_img, q_mask = split_ds._get_slice(query_idx)
        query_img = split_ds._resize(q_img, is_mask=False)
        query_mask = split_ds._resize(q_mask > 0, is_mask=True)

        supp_imgs_list = []
        supp_masks_list = []
        for si in support_indices:
            s_img, s_mask = split_ds._get_slice(si)
            supp_imgs_list.append(split_ds._resize(s_img, is_mask=False))
            supp_masks_list.append(split_ds._resize(s_mask > 0, is_mask=True))

        support_imgs = torch.stack(supp_imgs_list, dim=0)   # [S, C, H, W]
        support_masks = torch.stack(supp_masks_list, dim=0) # [S, 1, H, W]

        return query_img, query_mask, support_imgs, support_masks, ds_name

    def __getitem__(self, idx: int):
        return self.sample_episode(split="train")


def evaluate_dataset_split(model, dataloader, criterion, device, max_steps: int = None):
    """
    Evaluates model on a specific dataset split DataLoader.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            query_img, query_mask, support_imgs, support_masks = batch[:4]
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

            if max_steps is not None and (step + 1) >= max_steps:
                break

    mean_loss = running_loss / total_samples if total_samples > 0 else 0.0
    mean_dice = running_dice / total_samples if total_samples > 0 else 0.0
    return mean_loss, mean_dice


def load_baseline_scores(dataset: str):
    """
    Loads UniverSeg baseline Dice, Per-Dataset Trained Dice, and Previous Zero-Shot Dice.
    """
    universeg_dice = None
    per_dataset_dice = None
    prev_zeroshot_dice = None

    # 1. UniverSeg baseline score
    baseline_path = os.path.join("logs", f"{dataset}_baseline_scores.txt")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            for line in f:
                if "Mean Dice Score:" in line:
                    try:
                        universeg_dice = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass

    # 2. Per-dataset trained score
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

    # 3. Previous zero-shot score
    zeroshot_path = os.path.join("logs", f"zeroshot_{dataset}_scores.txt")
    if os.path.exists(zeroshot_path):
        with open(zeroshot_path, "r") as f:
            for line in f:
                if "Test Dice:" in line:
                    try:
                        prev_zeroshot_dice = float(line.split(":")[-1].strip())
                        break
                    except ValueError:
                        pass

    return universeg_dice, per_dataset_dice, prev_zeroshot_dice


def train_episodic(
    num_epochs: int = 30,
    episodes_per_epoch: int = 500,
    num_layers: int = 3,
    lr: float = 1e-3,
    num_support: int = 2,
    seed: int = 42,
    dry_run: bool = False,
    device: torch.device = None,
    train_datasets: list = None,
    checkpoint_path: str = "models/checkpoints/best_fusion_episodic.pt",
    log_file_path: str = "logs/episodic_training.txt",
):
    """
    Trains the In-Context Segmentation Model via Episodic Meta-Learning.
    """
    if dry_run:
        num_epochs = 2
        episodes_per_epoch = 50
        print(f"[DRY RUN] Overriding settings: {num_epochs} epochs, {episodes_per_epoch} episodes/epoch.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("==================================================")
    print(" Starting Episodic Meta-Learning Training")
    print("==================================================")
    print(f" Device:              {device}")
    print(f" Epochs:              {num_epochs}")
    print(f" Episodes per Epoch:  {episodes_per_epoch}")
    print(f" Num Layers (Depth):  {num_layers}")
    print(f" Learning Rate:       {lr}")
    print(f" Support Slices:      {num_support}")
    print(f" Seed:                {seed}")
    print(f" Training Datasets:   {train_datasets or SUPPORTED_DATASETS}")
    print(f" Checkpoint Target:   {checkpoint_path}")
    print(f" Dry Run Mode:        {dry_run}")
    print("--------------------------------------------------")

    dataset = EpisodicMedSegDataset(
        num_support=num_support,
        target_size=(128, 128),
        channel_idx=0,
        seed=seed,
        datasets=train_datasets,
    )

    # Initialize model with frozen UniverSeg backbone
    feature_channels = [64] * num_layers
    model = InContextSegmentationModel(
        in_channels=1,
        feature_channels=feature_channels,
        num_heads=4,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    model.print_parameter_summary()

    # Trainable parameters: FusionModule + Decoder only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.5)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    best_avg_val_dice = -1.0
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_dice_per_dataset": {d: [] for d in dataset.dataset_names},
        "val_avg_dice": [],
    }

    # Prepare Validation DataLoaders for each training dataset
    val_loaders = {}
    eval_batch_size = 16
    for d in dataset.dataset_names:
        val_loaders[d] = DataLoader(
            dataset.splits[d]["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

    val_max_steps = 2 if dry_run else None
    start_total_time = time.time()

    with open(log_file_path, "w") as log_f:
        log_f.write("Episodic Meta-Learning Training Log\n")
        log_f.write("==================================================\n")
        log_f.write(f"Epochs: {num_epochs}, Episodes/Epoch: {episodes_per_epoch}, LR: {lr}, Support: {num_support}, DryRun: {dry_run}\n\n")
        header = "Epoch,TrainLoss,TrainDice," + ",".join([f"ValDice_{d}" for d in dataset.dataset_names]) + ",AvgValDice\n"
        log_f.write(header)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_dice = 0.0
        organ_counts = {d: 0 for d in dataset.dataset_names}

        epoch_start_time = time.time()

        for ep_idx in range(episodes_per_epoch):
            q_img, q_mask, s_imgs, s_masks, organ = dataset.sample_episode(split="train")
            organ_counts[organ] += 1

            # Shape: [1, C, H, W] and [1, S, C, H, W]
            q_img = q_img.unsqueeze(0).to(device)
            q_mask = q_mask.unsqueeze(0).to(device)
            s_imgs = s_imgs.unsqueeze(0).to(device)
            s_masks = s_masks.unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(q_img, s_imgs, s_masks)
            loss = criterion(logits, q_mask)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                dice = compute_dice_score(preds, q_mask)

            running_train_loss += loss.item()
            running_train_dice += dice

        epoch_train_loss = running_train_loss / episodes_per_epoch
        epoch_train_dice = running_train_dice / episodes_per_epoch
        history["train_loss"].append(epoch_train_loss)
        history["train_dice"].append(epoch_train_dice)

        # Validation on each dataset separately
        val_dices = {}
        for d in dataset.dataset_names:
            _, v_dice = evaluate_dataset_split(
                model, val_loaders[d], criterion, device, max_steps=val_max_steps
            )
            val_dices[d] = v_dice
            history["val_dice_per_dataset"][d].append(v_dice)

        avg_val_dice = float(np.mean(list(val_dices.values())))
        history["val_avg_dice"].append(avg_val_dice)

        is_best = avg_val_dice > best_avg_val_dice
        if is_best:
            best_avg_val_dice = avg_val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_val_dice": best_avg_val_dice,
                    "val_dices": val_dices,
                    "history": history,
                    "train_datasets": dataset.dataset_names,
                },
                checkpoint_path,
            )

        epoch_time = time.time() - epoch_start_time
        val_str = " | ".join([f"{d[:4].capitalize()}: {val_dices[d]:.4f}" for d in dataset.dataset_names])

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:4.1f}s) "
            f"Loss: {epoch_train_loss:.4f} | Train Dice: {epoch_train_dice:.4f} || "
            f"{val_str} | AvgVal: {avg_val_dice:.4f}"
            f"{' [SAVED BEST]' if is_best else ''}"
        )

        with open(log_file_path, "a") as log_f:
            row = (
                f"{epoch},{epoch_train_loss:.4f},{epoch_train_dice:.4f},"
                + ",".join([f"{val_dices[d]:.4f}" for d in dataset.dataset_names])
                + f",{avg_val_dice:.4f}\n"
            )
            log_f.write(row)

    total_time = time.time() - start_total_time
    print("==================================================")
    print(f" Episodic Training Completed in {total_time:.1f} seconds.")
    print(f" Best Average Validation Dice: {best_avg_val_dice:.4f}")
    print(f" Saved Best Checkpoint To:     {checkpoint_path}")
    print("==================================================")

    return model, dataset, checkpoint_path


def evaluate_zeroshot_episodic(
    checkpoint_path: str = "models/checkpoints/best_fusion_episodic.pt",
    test_datasets: list = None,
    num_support: int = 2,
    seed: int = 42,
    dry_run: bool = False,
    device: torch.device = None,
    results_path: str = "logs/episodic_zeroshot_results.txt",
):
    """
    Evaluates the episodic meta-learning checkpoint across held-out test sets.
    Compares performance against UniverSeg baseline, per-dataset trained, and prior zero-shot results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_datasets = test_datasets if test_datasets is not None else list(SUPPORTED_DATASETS)

    dataset = EpisodicMedSegDataset(
        num_support=num_support,
        target_size=(128, 128),
        channel_idx=0,
        seed=seed,
        datasets=eval_datasets,
    )

    model = InContextSegmentationModel(
        in_channels=1,
        feature_channels=[64, 64, 64],
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Episodic checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n[INFO] Loaded episodic meta-learning checkpoint from {checkpoint_path}")

    criterion = BCEDiceLoss(bce_weight=0.5)
    test_max_steps = 2 if dry_run else None
    eval_batch_size = 16

    results = {}
    print("\n==========================================================================================")
    print(" HELD-OUT TEST EVALUATION (EPISODIC META-LEARNING CHECKPOINT)")
    print("==========================================================================================")

    for ds_name in eval_datasets:
        test_ds = dataset.splits[ds_name]["test"]
        test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)

        test_loss, episodic_dice = evaluate_dataset_split(
            model, test_loader, criterion, device, max_steps=test_max_steps
        )
        u_base, p_train, prev_zs = load_baseline_scores(ds_name)

        results[ds_name] = {
            "episodic_dice": episodic_dice,
            "test_loss": test_loss,
            "universeg_baseline": u_base,
            "per_dataset_trained": p_train,
            "prev_zeroshot": prev_zs,
        }

    # Print Table
    header = (
        f"{'Dataset':<12} | {'Episodic Dice':<14} | {'UniverSeg Base':<14} | "
        f"{'Diff vs US':<12} | {'Prev Zero-Shot':<14} | {'Diff vs Prev':<12} | "
        f"{'Per-DS Trained':<14} | {'Diff vs Per-DS':<14}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    ep_dices, us_dices, pd_dices, pz_dices = [], [], [], []

    for ds_name, r in results.items():
        ep_d = r["episodic_dice"]
        ep_dices.append(ep_d)

        us_d = r["universeg_baseline"]
        diff_us = f"{ep_d - us_d:+.4f}" if us_d is not None else "N/A"
        if us_d is not None:
            us_dices.append(us_d)

        pz_d = r["prev_zeroshot"]
        diff_pz = f"{ep_d - pz_d:+.4f}" if pz_d is not None else "N/A"
        if pz_d is not None:
            pz_dices.append(pz_d)

        pd_d = r["per_dataset_trained"]
        diff_pd = f"{ep_d - pd_d:+.4f}" if pd_d is not None else "N/A"
        if pd_d is not None:
            pd_dices.append(pd_d)

        us_str = f"{us_d:.4f}" if us_d is not None else "N/A"
        pz_str = f"{pz_d:.4f}" if pz_d is not None else "N/A"
        pd_str = f"{pd_d:.4f}" if pd_d is not None else "N/A"

        print(
            f"{ds_name:<12} | {ep_d:<14.4f} | {us_str:<14} | "
            f"{diff_us:<12} | {pz_str:<14} | {diff_pz:<12} | "
            f"{pd_str:<14} | {diff_pd:<14}"
        )

    print(separator)
    avg_ep = np.mean(ep_dices) if ep_dices else 0.0
    avg_us = np.mean(us_dices) if us_dices else None
    avg_pd = np.mean(pd_dices) if pd_dices else None
    avg_pz = np.mean(pz_dices) if pz_dices else None

    diff_avg_us = f"{avg_ep - avg_us:+.4f}" if avg_us is not None else "N/A"
    diff_avg_pz = f"{avg_ep - avg_pz:+.4f}" if avg_pz is not None else "N/A"
    diff_avg_pd = f"{avg_ep - avg_pd:+.4f}" if avg_pd is not None else "N/A"

    us_avg_str = f"{avg_us:.4f}" if avg_us is not None else "N/A"
    pz_avg_str = f"{avg_pz:.4f}" if avg_pz is not None else "N/A"
    pd_avg_str = f"{avg_pd:.4f}" if avg_pd is not None else "N/A"

    print(
        f"{'AVERAGE':<12} | {avg_ep:<14.4f} | {us_avg_str:<14} | "
        f"{diff_avg_us:<12} | {pz_avg_str:<14} | {diff_avg_pz:<12} | "
        f"{pd_avg_str:<14} | {diff_avg_pd:<14}"
    )
    print("==========================================================================================")

    # Save to file
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("Episodic Meta-Learning Zero-Shot Evaluation Results\n")
        f.write("==========================================================================================\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dry Run: {dry_run}\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for ds_name, r in results.items():
            ep_d = r["episodic_dice"]
            us_d = r["universeg_baseline"]
            diff_us = f"{ep_d - us_d:+.4f}" if us_d is not None else "N/A"
            pz_d = r["prev_zeroshot"]
            diff_pz = f"{ep_d - pz_d:+.4f}" if pz_d is not None else "N/A"
            pd_d = r["per_dataset_trained"]
            diff_pd = f"{ep_d - pd_d:+.4f}" if pd_d is not None else "N/A"
            us_str = f"{us_d:.4f}" if us_d is not None else "N/A"
            pz_str = f"{pz_d:.4f}" if pz_d is not None else "N/A"
            pd_str = f"{pd_d:.4f}" if pd_d is not None else "N/A"
            f.write(
                f"{ds_name:<12} | {ep_d:<14.4f} | {us_str:<14} | "
                f"{diff_us:<12} | {pz_str:<14} | {diff_pz:<12} | "
                f"{pd_str:<14} | {diff_pd:<14}\n"
            )
        f.write(separator + "\n")
        f.write(
            f"{'AVERAGE':<12} | {avg_ep:<14.4f} | {us_avg_str:<14} | "
            f"{diff_avg_us:<12} | {pz_avg_str:<14} | {diff_avg_pz:<12} | "
            f"{pd_avg_str:<14} | {diff_avg_pd:<14}\n"
        )
        f.write("==========================================================================================\n")

    print(f"\nSaved zero-shot evaluation summary to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Episodic Meta-Learning Training & Zero-Shot Evaluation for In-Context Segmentation"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--episodes_per_epoch", type=int, default=500, help="Episodes sampled per epoch")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of feature stages/layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_support", type=int, default=2, help="Support slices per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry run (2 epochs, 50 episodes)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run zero-shot evaluation only")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_fusion_episodic.pt")
    parser.add_argument("--train_datasets", nargs="+", default=None, choices=SUPPORTED_DATASETS, help="Datasets to train on")
    parser.add_argument("--test_dataset", type=str, default=None, choices=SUPPORTED_DATASETS, help="Specific held-out test dataset")
    parser.add_argument("--results_file", type=str, default="logs/episodic_zeroshot_results.txt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = args.train_datasets
    test_ds = [args.test_dataset] if args.test_dataset is not None else None

    # Validation: Held-out test dataset cannot be in train_datasets if explicitly doing leave-one-out
    if args.test_dataset and train_ds and args.test_dataset in train_ds:
        parser.error(f"Held-out test dataset '{args.test_dataset}' cannot be in --train_datasets ({train_ds})")

    if not args.eval_only:
        model, dataset, ckpt_path = train_episodic(
            num_epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            num_layers=args.num_layers,
            lr=args.lr,
            num_support=args.num_support,
            seed=args.seed,
            dry_run=args.dry_run,
            device=device,
            train_datasets=train_ds,
            checkpoint_path=args.checkpoint,
            log_file_path=f"logs/episodic_training_{args.test_dataset}.txt" if args.test_dataset else "logs/episodic_training.txt",
        )
        evaluate_zeroshot_episodic(
            checkpoint_path=ckpt_path,
            test_datasets=test_ds,
            num_support=args.num_support,
            seed=args.seed,
            dry_run=args.dry_run,
            device=device,
            results_path=args.results_file,
        )
    else:
        evaluate_zeroshot_episodic(
            checkpoint_path=args.checkpoint,
            test_datasets=test_ds,
            num_support=args.num_support,
            seed=args.seed,
            dry_run=args.dry_run,
            device=device,
            results_path=args.results_file,
        )


if __name__ == "__main__":
    main()

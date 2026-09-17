"""
evaluate_support_selection.py - Support Slice Selection Sensitivity Ablation
===========================================================================
Ablation study to evaluate the sensitivity of the In-Context Cross-Attention
Adapter to the choice of support exemplar slice:
  1. Max Area: Slice with the largest foreground mask area.
  2. Median Area: Slice nearest the median foreground mask area.
  3. Edge / Boundary: Slice with < 20% of max target area (small edge slice).
  4. Random: Randomly sampled positive slice.

Tests robustness to clinical user variability across Spleen CT and Heart MRI.
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    from src.models.in_context_model import InContextSegmentationModel
    from src.data_loading import get_dataloader
except ImportError as e:
    print(f"[WARN] Local imports: {e}")


def run_support_ablation(checkpoint_path: str, datasets: list, output_file: str, device: str = "cuda"):
    """Runs deterministic support slice selection strategies."""
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[INFO] Using device: {device_obj}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    model = InContextSegmentationModel(num_layers=3).to(device_obj)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device_obj)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        print("[INFO] Model weights loaded successfully.")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}, using init weights.")

    model.eval()

    strategies = ["max_area", "median_area", "edge_slice", "random"]
    ablation_results = {}

    for ds_name in datasets:
        print(f"\nEvaluating Support Slice Selection on {ds_name}...")
        ablation_results[ds_name] = {}

        for strat in strategies:
            print(f"  --> Strategy: {strat}...")
            try:
                test_loader = get_dataloader(ds_name, split="test", batch_size=4, num_workers=0)
            except Exception as e:
                print(f"  Could not load dataloader: {e}")
                continue

            dices = []
            with torch.no_grad():
                for batch in test_loader:
                    query = batch["query"].to(device_obj)
                    support_imgs = batch["support_imgs"].to(device_obj)
                    support_masks = batch["support_masks"].to(device_obj)
                    targets = batch["target"].to(device_obj)

                    # Deterministic support slice sorting based on strategy
                    B, S, C, H, W = support_masks.shape
                    mask_areas = support_masks.view(B, S, -1).sum(dim=-1)  # [B, S]

                    # Filter or sort slices according to strategy
                    if strat == "max_area":
                        _, top_idx = torch.topk(mask_areas, k=1, dim=-1)
                    elif strat == "edge_slice":
                        # Pick smallest non-zero mask area
                        zero_masked = mask_areas.clone()
                        zero_masked[zero_masked == 0] = 1e9
                        _, top_idx = torch.topk(zero_masked, k=1, dim=-1, largest=False)
                    elif strat == "median_area":
                        median_val = mask_areas.median(dim=-1, keepdim=True)[0]
                        diff = torch.abs(mask_areas - median_val)
                        _, top_idx = torch.topk(diff, k=1, dim=-1, largest=False)
                    else:  # random
                        top_idx = torch.randint(0, S, (B, 1), device=device_obj)

                    # Gather the chosen single support slice
                    chosen_img = torch.gather(support_imgs, 1, top_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, C, H, W))
                    chosen_mask = torch.gather(support_masks, 1, top_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 1, H, W))

                    preds = model(query, chosen_img, chosen_mask)
                    preds = (torch.sigmoid(preds) > 0.5).float()

                    # Compute dice
                    intersection = (preds * targets).sum(dim=(1, 2, 3))
                    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
                    dice_batch = (2.0 * intersection / (total + 1e-7)).cpu().numpy()
                    dices.extend(dice_batch)

            mean_d = float(np.mean(dices)) if len(dices) > 0 else 0.0
            std_d = float(np.std(dices)) if len(dices) > 0 else 0.0
            ablation_results[ds_name][strat] = (mean_d, std_d)
            print(f"      Mean Dice: {mean_d:.4f} +/- {std_d:.4f}")

    # Write report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("================================================================================\n")
        f.write("          SUPPORT SLICE SELECTION SENSITIVITY ABLATION REPORT\n")
        f.write("================================================================================\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"{'Target Dataset':<15} | {'Max Area (Optimal)':<22} | {'Median Area':<20} | {'Edge (<20%)':<20} | {'Random':<20}\n")
        f.write("-" * 105 + "\n")
        for ds, strats in ablation_results.items():
            max_s = f"{strats.get('max_area', (0,0))[0]:.4f} +/- {strats.get('max_area', (0,0))[1]:.4f}"
            med_s = f"{strats.get('median_area', (0,0))[0]:.4f} +/- {strats.get('median_area', (0,0))[1]:.4f}"
            edg_s = f"{strats.get('edge_slice', (0,0))[0]:.4f} +/- {strats.get('edge_slice', (0,0))[1]:.4f}"
            rnd_s = f"{strats.get('random', (0,0))[0]:.4f} +/- {strats.get('random', (0,0))[1]:.4f}"
            f.write(f"{ds:<15} | {max_s:<22} | {med_s:<20} | {edg_s:<20} | {rnd_s:<20}\n")
        f.write("=" * 105 + "\n")

    print(f"\n[DONE] Saved support selection ablation to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support Slice Selection Ablation")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_fusion_episodic.pt")
    parser.add_argument("--datasets", nargs="+", default=["spleen", "heart"])
    parser.add_argument("--output", type=str, default="logs/support_selection_ablation.txt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_support_ablation(args.checkpoint, args.datasets, args.output, args.device)
